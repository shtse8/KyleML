import os
from multiprocessing import Pool, TimeoutError
import __main__
import random
import numpy as np
import pandas as pd
from operator import add
import sys
import collections
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt
from memories.Transition import Transition
from utils.errors import InvalidAction
from enum import Enum
import torch
import torch.nn as nn
from games.Game import Game
import torch.multiprocessing as mp
import humanize
from utils.PredictionHandler import PredictionHandler


class Phrase(Enum):
    train = 1
    test = 2
    play = 3


class Agent(object):
    def __init__(self, name: str, env: Game, **kwargs) -> None:
        self.name: str = name

        # Model
        self.env: Game = env
        
        # Trainning
        self.target_epochs: int = kwargs.get('target_epochs', 10000)
        self.target_trains: int = kwargs.get('target_trains', 1000)
        self.target_tests: int = kwargs.get('target_tests', 100)
        
        self.target_episodes: int = 0
        
        # self.phrases = [Phrase.train, Phrase.test]
        self.phrases = []
        # self.phrases = [Phrase.play]

        # episode stats
        self.report = EpisodeReport()

        # phrase stats
        self.episodes = mp.Value('i', 0)
        
        # epoch stats
        self.phraseIndex: int = 0

        # training stats
        self.epochs: int = 0
        self.totalSteps: int = 0

        # History
        self.history = None
        
        self.startTime = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Save And Load
        self.weightPath = kwargs.get('weightPath', "./weights/")
        self.models = []

        # plt.ion()
        plt.ylabel("Loss")
        plt.xlabel("Episode")
        # plt.show(block=False)

    def addModels(self, model: nn.Module) -> None:
        self.models.append(model)

    def beginEpoch(self) -> None:
        self.epochs += 1
        self.phraseIndex = 0
        return self.epochs <= self.target_epochs

    def endEpoch(self) -> None:
        pass

    def getPrediction(self, state):
        raise NotImplementedError()

    def getAction(self, state, mask = None) -> int:
        raise NotImplementedError()

    def commit(self, transition: Transition) -> None:
        self.report.rewards += transition.reward
        # self.update()

    def run(self, train: bool = True) -> None:
        self.phrases = [Phrase.train] if train else [Phrase.play]
        self.start()

    def start(self):
        while self.beginEpoch():
            while self.beginPhrase():
                gen = self.getSample()
                while self.beginEpisode():
                    while True:
                        transition = next(gen)
                        self.commit(transition)
                        if transition.done:
                            break
                    self.endEpisode()
                self.endPhrase()
            self.endEpoch()
    
    def getSample(self):
        # samples = collections.deque(maxlen=num)
        while True:
            state = self.env.reset()
            done: bool = False
            while not done:
                reward: float = 0.
                nextState = state
                # actionMask = self.env.getActionMask(state)
                actionMask = np.ones(self.env.actionSpace)
                prediction = self.getPrediction(state)
                while True:
                    try:
                        # print(actionMask)
                        # print(state, self.env.getActionMask(), actionMask)
                        # if prediction.max() > 0.95 and actionMask[prediction.argmax()] == 0:
                        #     print(state, prediction, actionMask)
                        action = self.getAction(prediction, actionMask)
                        nextState, reward, done = self.env.takeAction(action)
                        transition = Transition(state, action, reward, nextState, done, prediction)
                        yield transition
                        # samples.append(transition)
                        # if len(samples) >= num or (endOnDone and done):
                        #     yield samples
                        #     samples.clear()
                        break
                    except InvalidAction:
                        # print(prediction, state, action, "Failed")
                        actionMask[action] = 0
                        self.report.invalidMoves += 1
                        yield Transition(state, action, 0, state, False, prediction)
                        # print(actionMask)
                    # finally:
                state = nextState

    def getPhrase(self) -> Phrase:
        return self.phrases[self.phraseIndex]

    def isTraining(self) -> bool:
        return self.getPhrase() == Phrase.train

    def isTesting(self) -> bool:
        return self.getPhrase() == Phrase.test

    def isPlaying(self) -> bool:
        return self.getPhrase() == Phrase.play

    def beginPhrase(self) -> bool:
        if self.phraseIndex >= len(self.phrases):
            return False
        self.episodes.value = 0

        self.startTime = time.perf_counter()
        if self.isTraining():
            self.target_episodes = self.target_trains
        elif self.isTesting():
            self.target_episodes = self.target_tests
        elif self.isPlaying():
            self.target_episodes = 100
        
        self.history = collections.deque(maxlen=self.target_episodes)
        return True

    def endPhrase(self) -> None:
        if self.isTraining():
            self.save()

        self.phraseIndex += 1
        # plt.cla()
        # plt.plot(self.lossHistory)
        # plt.show()
        # plt.draw()
        # plt.pause(0.00001)
        print()

    def beginEpisode(self) -> None:
        self.episodes.value += 1
        self.report = EpisodeReport()
        return self.episodes.value <= self.target_episodes

    def endEpisode(self) -> None:
        self.totalSteps += self.report.steps
        self.history.append(self.report)
        self.update()

    def update(self) -> None:
        duration = time.perf_counter() - self.startTime
        avgLoss = np.mean([x.loss for x in self.history]) if len(self.history) > 0 else math.nan
        bestReward = np.max([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        avgReward = np.mean([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        stdReward = np.std([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        progress = self.episodes.value / self.target_episodes
        invalidMovesPerEpisode = np.mean([x.invalidMoves for x in self.history])
        durationPerEpisode = duration / self.episodes.value
        estimateDuration = self.target_episodes * durationPerEpisode
        totalSteps = np.sum([x.steps for x in self.history])
        print(f"{self.getPhrase().name:5} #{self.epochs} {progress:>4.0%} {humanize.intword(self.totalSteps)} | " + \
            f'Loss: {avgLoss:6.2f}/ep | Best: {bestReward:>5}, Avg: {avgReward:>5.2f} | Steps: {totalSteps/duration:>7.2f}/s | Episodes: {1/durationPerEpisode:>6.2f}/s | Invalid: {invalidMovesPerEpisode: >6.2f}/ep | Time: {duration: >4.2f}s > {estimateDuration: >5.2f}s', end = "\b\r")

    def learn(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        try:
            path = self.getSavePath(True)
            data = {
                "totalSteps": self.totalSteps
            }
            for model in self.models:
                data[model.name] = model.state_dict()
            torch.save(data, path)
            # print("Saved Weights.")
        except Exception as e:
            print("Failed to save.", e)
        
    def load(self) -> None:
        try:
            path = self.getSavePath()
            print("Loading from path: ", path)
            data = torch.load(path, map_location=self.device)
            self.totalSteps = int(data["totalSteps"]) if "totalSteps" in data else 0
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
        self.invalidMoves: int = 0
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
        return self.total_loss / self.steps if self.steps > 0 else math.nan

    def trained(self, loss, steps):
        self.total_loss += loss * steps
        self.steps += steps
        