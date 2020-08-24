import os
import __main__
import numpy as np
import collections
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
from memories.Transition import Transition
from enum import Enum
import torch
import torch.nn as nn
from games.Game import Game
import torch.multiprocessing as mp


class Mode(Enum):
    train = 1
    eval = 1


class Agent(object):
    def __init__(self, name: str, env: Game, **kwargs) -> None:
        self.name: str = name

        # Model
        self.env: Game = env
        
        # Trainning
        self.mode = Mode.train
        self.delay: int = 0
        self.target_epochs: int = 0
        self.target_episodes: int = 0
        
        # training stats
        self.episodes = mp.Value('i', 0)
        self.epochs: int = 0
        self.totalEpisodes: int = 0
        self.totalSteps: int = 0
        
        # History
        self.history = None
        
        self.startTime = 0
        self.lastSave = 0
        self.lastPrint = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Save And Load
        self.weightPath = kwargs.get('weightPath', "./weights/")
        self.models = []

        # multithread
        self.queue = mp.Queue()

        # plt.ion()
        plt.ylabel("Loss")
        plt.xlabel("Episode")
        # plt.show(block=False)

    def addModels(self, model: nn.Module) -> None:
        self.models.append(model)

    def epochBegin(self) -> None:
        self.epochs += 1
        self.episodes.value = 0
        self.startTime = time.perf_counter()
        self.history = collections.deque(maxlen=self.target_episodes)

        return self.epochs <= self.target_epochs

    def epochEnd(self) -> None:
        # plt.cla()
        # plt.plot(self.lossHistory)
        # plt.show()
        # plt.draw()
        # plt.pause(0.00001)
        print()

    def getPrediction(self, state):
        raise NotImplementedError()

    def getAction(self, state, mask=None) -> int:
        raise NotImplementedError()

    def run(self, train: bool = True, episodes: int = 1000, epochs: int = 10000, delay: float = 0) -> None:
        self.delay = delay
        self.mode = Mode.train if train else Mode.eval
        self.target_episodes = episodes
        self.target_epochs = epochs
        self.start()

    def onEpochBegin(self):
        pass

    def onEpisodeBegin(self):
        pass

    def onCommit(self, transition: Transition):
        pass

    def onReceive(self, memory):
        # receive memory from child
        self.learn(memory)

    def onInit(self):
        pass

    def mainWorker(self):
        print("Main Worker Started")
        self.onInit()
        self.epochBegin()
        while True:
            try:
                memory = self.queue.get_nowait()
                # print(self.queue.qsize(), len(memory))
                self.episodeBegin()
                self.learn(memory)
                for network in self.models:
                    print(network.state_dict())
                self.episodeEnd()
            except Exception:
                pass
            if time.perf_counter() - self.lastPrint > 1:
                self.update()

    def childWorker(self, i):
        print("Child Worker Started")
        self.onInit()
        self.report = EpisodeReport()
        self.onEpochBegin()
        while True:
            self.onEpisodeBegin()
            state = self.env.reset()
            done: bool = False
            while not done:
                actionMask = np.ones(self.env.actionSpace)
                prediction = self.getPrediction(state)
                action = self.getAction(prediction, actionMask)
                nextState, reward, done = self.env.takeAction(action)
                transition = Transition(state, action, reward, nextState, done, prediction)
                self.commit(transition)
                if self.delay > 0:
                    time.sleep(self.delay)
                state = nextState

    def commit(self, transition: Transition):
        self.report.rewards += transition.reward
        self.onCommit(transition)

    def start(self):
        print(f"Mode: {self.mode.name}, Total Episodes: {self.totalEpisodes}, Total Steps: {self.totalSteps}")
        processes = []
        n_workers = 1 # mp.cpu_count() // 2
        for i in range(n_workers):
            p = mp.Process(target=self.childWorker, args=(i,))
            p.start()
            processes.append(p)
        
        self.mainWorker()

        """
        while self.epochBegin():
            self.onEpochBegin()
            while self.episodeBegin():
                self.onEpisodeBegin()
                state = self.env.reset()
                done: bool = False
                while not done:
                    reward = 0.
                    nextState = state
                    # actionMask = self.env.getActionMask(state)
                    actionMask = np.ones(self.env.actionSpace)
                    prediction = self.getPrediction(state)
                    while True:
                        action = self.getAction(prediction, actionMask)
                        nextState, reward, done = self.env.takeAction(action)
                        break
                        if not (state == nextState).all():
                            break
                        actionMask[action] = 0
                    transition = Transition(state, action, reward, nextState, done, prediction)
                    self.commit(transition)
                    if self.delay > 0:
                        time.sleep(self.delay)
                    state = nextState
                self.episodeEnd()
            self.epochEnd()
        """
    def isTraining(self) -> bool:
        return self.mode == Mode.train

    def isEval(self) -> bool:
        return self.mode == Mode.eval

    def episodeBegin(self) -> None:
        self.episodes.value += 1
        self.report = EpisodeReport()
        self.history.append(self.report)
        return self.episodes.value <= self.target_episodes

    def episodeEnd(self) -> None:
        if self.isTraining():
            self.totalEpisodes += 1
            self.totalSteps += self.report.steps
            if time.perf_counter() - self.lastSave > 60:
                self.save()

    def update(self) -> None:
        duration = time.perf_counter() - self.startTime
        avgLoss = np.mean([x.loss for x in self.history]) if len(self.history) > 0 else math.nan
        bestReward = np.max([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        avgReward = np.mean([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        # stdReward = np.std([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        progress = self.episodes.value / self.target_episodes
        # invalidMovesPerEpisode = np.mean([x.invalidMoves for x in self.history])
        durationPerEpisode = duration / self.episodes.value if self.episodes.value > 0 else math.nan
        estimateDuration = self.target_episodes * durationPerEpisode
        totalSteps = np.sum([x.steps for x in self.history])
        print(f"#{self.epochs} {progress:>4.0%} {humanize.intword(self.totalSteps)} | " +
              f'Loss: {avgLoss:6.2f}/ep | Best: {bestReward:>5}, Avg: {avgReward:>5.2f} | ' +
              f'Steps: {totalSteps/duration:>7.2f}/s | Episodes: {1/durationPerEpisode:>6.2f}/s | ' +
              f'Time: {duration: >4.2f}s > {estimateDuration: >5.2f}s', end="\b\r")
        self.lastPrint = time.perf_counter()

    def learn(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        try:
            path = self.getSavePath(True)
            data = {
                "totalSteps": self.totalSteps,
                "totalEpisodes": self.totalEpisodes
            }
            for model in self.models:
                data[model.name] = model.state_dict()
            torch.save(data, path)
            self.lastSave = time.perf_counter()
            # print("Saved Weights.")
        except Exception as e:
            print("Failed to save.", e)
        
    def load(self) -> None:
        try:
            path = self.getSavePath()
            print("Loading from path: ", path)
            data = torch.load(path, map_location=self.device)
            self.totalSteps = int(data["totalSteps"]) if "totalSteps" in data else 0
            self.totalEpisodes = int(data["totalEpisodes"]) if "totalEpisodes" in data else 0
            for model in self.models:
                model.load_state_dict(data[model.name])
            print("Weights loaded.")
        except Exception as e:
            print("Failed to load.", e)
    
    def getSavePath(self, makeDir: bool = False) -> str:
        path = os.path.join(os.path.dirname(__main__.__file__), self.weightPath, self.name, self.env.name + ".h5")
        if makeDir:
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        return path


class EpisodeManager:
    def __init__(self, target):
        self.count = 0
        self.target = target

    def next(self):
        if self.count > 0:
            self.end()
        self.start()
        return self.count < self.self.target

    def end(self):
        pass

    def start(self):
        self.count += 1


class Message:
    def __init__(self):
        pass


class EpisodeReport(Message):
    def __init__(self):
        self.steps: int = 0
        self.rewards: float = 0
        self.total_loss: float = 0
        self.episode_start_time: int = 0
        self.episode_end_time: int = 0

    def start(self):
        self.episode_start_time = time.perf_counter()

    def end(self):
        self.episode_end_time = time.perf_counter()

    @property
    def duration(self):
        return (self.episode_end_time if self.episode_end_time > 0 else time.perf_counter()) - self.episode_start_time

    @property
    def loss(self):
        return self.total_loss / self.steps if self.steps > 0 else 0

    def trained(self, loss, steps):
        self.total_loss += loss * steps
        self.steps += steps
        