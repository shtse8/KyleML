import os
import __main__
import numpy as np
import collections
import math
import inspect
import time
from enum import Enum
import asyncio

from pathlib import Path
import matplotlib.pyplot as plt
from memories.Transition import Transition
import torch
import torch.nn as nn
from games.Game import Game
from typing import List, Callable, TypeVar, Generic, Tuple, Any
import torch.multiprocessing as mp
from multiprocessing.managers import NamespaceProxy, SyncManager
from multiprocessing.connection import Pipe
from games.GameFactory import GameFactory
import utils.Function as Function
from utils.Message import NetworkInfo, LearnReport, EnvReport, MethodCallRequest, MethodCallResult
from utils.PipedProcess import Process, PipedProcess
from utils.multiprocessing import Proxy
from utils.Network import Network
from utils.Normalizer import RangeNormalizer, StdNormalizer
from utils.KyleList import KyleList
from .mcts import MCTS


class Action:
    def __init__(self, index, mask, prediction):
        self.index = index
        self.mask = mask
        self.prediction = prediction

    def __int__(self):
        return self.index

    @property
    def log(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py#L72
        eps = torch.finfo(torch.float).eps
        prob = np.array([p if self.mask[i] else 0 for i,
                         p in enumerate(self.prediction)])
        prob = prob / prob.sum()
        prob = min(1-eps, max(eps, self.prediction[self.index]))
        return math.log(prob)

C = TypeVar('C')

class Algo(Generic[C]):
    def __init__(self, name, config: C):
        self.name = name
        self.config = config
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def createNetwork(self, inputShape, n_outputs) -> Network:
        raise NotImplementedError

    def getAction(self, network, state, mask, isTraining: bool, hiddenState=None) -> Tuple[Action, Any]:
        raise NotImplementedError

    def learn(self, network: Network, memory):
        raise NotImplementedError



class EpochManager:
    def __init__(self):
        self.num = 0
        self.epoch = None
        self.history = []
        self.eventHandlers = {}

    def on(self, eventName, handler):
        if eventName not in self.eventHandlers:
            self.eventHandlers[eventName] = []
        self.eventHandlers[eventName].append(handler)

    def emit(self, eventName):
        if eventName not in self.eventHandlers:
            return
        for handler in self.eventHandlers[eventName]:
            handler()

    def start(self, num):
        self.epoch = Epoch().start(num)
        self.num += 1
        return self

    def restart(self):
        self.history.append(self.epoch)
        self.emit("restart")
        self.start(self.epoch.target_episodes)
        return self

    def add(self, reports):
        self.epoch.add(reports)
        self.emit("add")
        return self

    def trained(self, loss, steps):
        self.epoch.trained(loss, steps)
        self.emit("trained")
        if self.epoch.isEnd:
            self.restart()
        return self



class Epoch:
    def __init__(self):
        self.target_episodes = 0
        self.steps: int = 0
        self.drops: int = 0
        self.rewards: float = 0
        self.avgLoss: float = 0
        self.total_loss: float = 0
        self.epoch_start_time: int = 0
        self.epoch_end_time: int = 0
        self.episodes = 0

        # for stats
        # self.history = collections.deque(maxlen=target_episodes)
        self.bestRewards = -math.inf
        self._avgRewards: float = 0
        # self.totalRewards = 0
        self.envs = 0

    def start(self, target_episodes):
        self.epoch_start_time = time.perf_counter()
        self.target_episodes = target_episodes
        return self

    def end(self):
        self.epoch_end_time = time.perf_counter()
        return self

    @property
    def hitRate(self):
        return self.steps / (self.steps + self.drops) if (self.steps + self.drops) > 0 else math.nan

    @property
    def isEnd(self):
        return self.epoch_end_time > 0

    @property
    def progress(self):
        return self.episodes / self.target_episodes

    @property
    def duration(self):
        return (self.epoch_end_time if self.epoch_end_time > 0 else time.perf_counter()) - self.epoch_start_time

    @property
    def loss(self):
        return self.avgLoss if self.steps > 0 else math.nan

    @property
    def durationPerEpisode(self):
        return self.duration / self.episodes if self.episodes > 0 else math.inf

    @property
    def estimateDuration(self):
        return self.target_episodes * self.durationPerEpisode

    @property
    def avgRewards(self):
        return self._avgRewards if self.envs > 0 else math.nan

    def add(self, reports):
        for report in reports:
            if report.rewards > self.bestRewards:
                self.bestRewards = report.rewards
            # self.totalRewards += report.rewards
            self.envs += 1
            self._avgRewards += (report.rewards - self._avgRewards) / self.envs
            # self.history.append(report)
        return self

    def trained(self, loss, steps):
        # self.total_loss += loss * steps
        self.steps += steps
        self.avgLoss += (loss - self.avgLoss) * steps / self.steps
        self.episodes += 1
        if self.episodes >= self.target_episodes:
            self.end()
        return self

class Config:
    def __init__(self, sampleSize=512, batchSize=32, learningRate=3e-4):
        self.sampleSize = sampleSize
        self.batchSize = batchSize
        self.learningRate = learningRate


class Base:
    def __init__(self, algo, gameFactory, sync):
        self.algo = algo
        self.algo.device = sync.getDevice()
        self.gameFactory = gameFactory
        self.sync = sync
        self.weightPath = "./weights/"
        self.lastSave = 0
        self.handler = None

    def save(self) -> None:
        try:
            path = self.getSavePath(True)
            data = {
                "totalSteps": self.sync.totalSteps.value,
                "totalEpisodes": self.sync.totalEpisodes.value
            }
            if self.handler is not None:
                handlerData = self.handler.dump()
                data = {**data, **handlerData}
            torch.save(data, path)
            self.lastSave = time.perf_counter()
            # print("Saved Weights.")
        except Exception as e:
            print("Failed to save.", e)

    def load(self) -> None:
        try:
            path = self.getSavePath()
            print("Loading from path: ", path)
            data = torch.load(path, map_location='cpu')
            # data = torch.load(path, map_location=self.device)
            self.sync.totalSteps.value = int(
                data["totalSteps"]) if "totalSteps" in data else 0
            self.sync.totalEpisodes.value = int(
                data["totalEpisodes"]) if "totalEpisodes" in data else 0
            if self.handler is not None:
                self.handler.load(data)
            print(
                f"Trained: {Function.humanize(self.sync.totalEpisodes.value)} episodes, {Function.humanize(self.sync.totalSteps.value)} steps")
        except Exception as e:
            print("Failed to load.", e)

    def getSavePath(self, makeDir: bool = False) -> str:
        path = os.path.join(
            self.weightPath, self.algo.name.lower(), self.gameFactory.name + ".h5")
        if makeDir:
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        return path



class Promise:
    def __init__(self):
        self.result = None


class Service(PipedProcess):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory
        self.isRunning = True
        self.callPipes = Pipe(True)
        self.eventPipes = Pipe(False)

    async def asyncRun(self, conn):
        # print("Evaluator", os.getpid(), conn)
        self.object = self.factory()
        while self.isRunning:
            if self.callPipes[1].poll():
                message = self.callPipes[1].recv()
                if isinstance(message, MethodCallRequest):
                    # print("MMethodCallRequest", message.method)
                    result = getattr(self.object, message.method)(
                        *message.args)
                    if inspect.isawaitable(result):
                        result = await result
                    self.callPipes[1].send(MethodCallResult(result))
            await asyncio.sleep(0)

    async def _waitResponse(self, future):
        while not self.callPipes[0].poll():
            await asyncio.sleep(0)
        message = self.callPipes[0].recv()
        future.set_result(message)

    def call(self, method, args=()):
        # print("Call", method)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.callPipes[0].send(MethodCallRequest(method, args))
        loop.create_task(self._waitResponse(future))
        return future


class EvaluatorService(Service):
    def __init__(self, algo, gameFactory, sync):
        self.algo = algo
        self.gameFactory = gameFactory
        self.sync = sync
        super().__init__(self.factory)

    def factory(self):
        return Evaluator(self.algo, self.gameFactory, self.sync)


class TrainerProcess(Process):
    def __init__(self, algo, gameFactory, sync, episodes, load):
        super().__init__()
        self.algo = algo
        self.gameFactory = gameFactory
        self.sync = sync
        self.episodes = episodes
        self.load = load

    async def asyncRun(self):
        # print("Trainer", os.getpid())
        await Trainer(self.algo, self.gameFactory, self.sync).start(self.episodes, self.load)

class SyncContext:
    # EpochProxy = Proxy(Epoch)
    EpochManagerProxy = Proxy(EpochManager)

    def __init__(self):
        # manager = mp.Manager()
        # SyncManager.register('Epoch', Epoch, self.EpochProxy)
        SyncManager.register('EpochManager', EpochManager, self.EpochManagerProxy)
        manager = SyncManager()
        manager.start()

        self.latestStateDict = manager.dict()
        self.latestVersion = manager.Value('i', -1)
        self.deviceIndex = manager.Value('i', 0)
        self.totalEpisodes = manager.Value('i', 0)
        self.totalSteps = manager.Value('i', 0)
        self.epochManager = manager.EpochManager()
        # self.epoch = manager.Epoch()

    def getDevice(self) -> torch.device:
        deviceName = "cpu"
        if torch.cuda.is_available():
            cudaId = self.deviceIndex.value % torch.cuda.device_count()
            deviceName = "cuda:" + str(cudaId)
            self.deviceIndex.value = self.deviceIndex.value + 1
        return torch.device(deviceName)


class Role(Enum):
    Trainer = 1
    Evaluator = 2

class Trainer(Base):
    def __init__(self, algo: Algo, gameFactory: GameFactory, sync):
        super().__init__(algo, gameFactory, sync)
        self.evaluators: List[EvaluatorService] = []
        self.handler = None
        self.rewardNormalizer = StdNormalizer()

    def learn(self, experience):
        steps = len(experience)
        loss = self.handler.learn(experience)
        # learn report handling
        self.sync.epochManager.trained(loss, steps)
        self.sync.totalEpisodes.value += 1
        self.sync.totalSteps.value += steps

    def pushNewNetwork(self):
        data = self.handler.dump()
        self.sync.latestStateDict.update(data)

    async def start(self, episodes=1000, load=False):
        env = self.gameFactory.get()

        # Create Evaluators
        evaluators = []
        n_workers = max(torch.cuda.device_count(), 1)
        for _ in range(n_workers):
            evaluator = EvaluatorService(self.algo, self.gameFactory, self.sync).start()
            evaluators.append(evaluator)

        self.evaluators = np.array(evaluators)

        self.handler = self.algo.createHandler(env, Role.Trainer, self.sync.getDevice())
        if load:
            self.load()
        n_samples = self.algo.config.sampleSize * n_workers
        evaulator_samples = self.algo.config.sampleSize

        self.sync.epochManager.start(episodes)
        self.lastSave = time.perf_counter()
        while True:
            # push new network
            self.pushNewNetwork()
            # collect samples
            experience = collections.deque(maxlen=n_samples)
            promises = np.array([x.call("roll", (evaulator_samples,))
                                 for x in self.evaluators])
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed
            for promise in asyncio.as_completed(promises):
                response = await promise  # earliest result
                for e in response.result:
                    # print(type(e))
                    e = KyleList[Transition](e)
                    self.handler.preprocess(e)
                    experience.extend(e)
            # print("learn")
            # learn
            experience = KyleList[Transition](experience)
            self.learn(experience)

            if time.perf_counter() - self.lastSave > 60:
                self.save()



class Evaluator(Base):
    def __init__(self, algo: Algo, gameFactory, sync):
        super().__init__(algo, gameFactory, sync)
        self.env = gameFactory.get()
        # self.algo.device = torch.device("cpu")
        # self.algo.device = sync.getDevice()
        self.handler = self.algo.createHandler(self.env, Role.Evaluator, sync.getDevice())

        # self.network = self.algo.createNetwork(
        #     self.env.observationShape, self.env.actionSpace).to(self.algo.device)
        # self.network.version = -1
        # self.networks.append(self.network)

        self.playerCount = self.env.getPlayerCount()
        # self.mcts = MCTS(lambda s, m, x: self.algo.getProb(self.network, s, m, x), n_playout=100)
        self.agents = np.array(
            [Agent(i + 1, self.env, self.handler) for i in range(self.playerCount)],)
        self.started = False

        self.reports: List[EnvReport] = []

    def updateNetwork(self):
        self.handler.load(self.sync.latestStateDict)

    def loop(self, num=0):
        # auto reset
        if not self.started:
            self.env.reset()
            self.started = True
        elif self.env.isDone():
            # reports = []
            for agent in self.agents:
                self.reports.append(agent.done())
            self.env.reset()
            self.handler.reset()
            return False

        return True
        # memoryCount = min([len(x.memory) for x in self.agents])
        if num > 0:
            memoryCount = min([len(x.memory) for x in self.agents])
            return memoryCount < num
        else:
            return True

    def flushReports(self):
        if len(self.reports) > 0:
            self.sync.epochManager.add(self.reports)
            self.reports = []

    def roll(self, num):
        num = 10000
        self.updateNetwork()
        self.generateTransitions(num)
        return [x.memory for x in self.agents]

    def generateTransitions(self, num):
        for agent in self.agents:
            agent.resetMemory(num)
        while self.loop(num):
            for agent in self.agents:
                agent.step(True)
        self.flushReports()

    async def eval(self):
        self.load()
        self.sync.epochManager.start(1)
        while self.loop():
            for agent in self.agents:
                if agent.id == 1:
                    player = self.env.getPlayer(agent.id)
                    if not self.env.isDone() and player.canStep():
                        while True:
                            try:
                                # row, col = map(int, input(
                                #     "Your action: ").split())
                                # pos = row * self.env.size + col
                                pos = int(input("Your action: "))
                                player.step(pos)
                                self.handler.reportStep(pos)
                                break
                            except Exception as e:
                                print(e)
                else:
                    agent.step(False)
                # os.system('cls')
                state = self.env.getState(1).astype(int)
                print(state[0] + state[1] * 2, "\n")
                await asyncio.sleep(0.5)
            if self.env.isDone():
                print("Done:")
                for agent in self.agents:
                    print(agent.id, self.env.getDoneReward(agent.id))
                await asyncio.sleep(3)
            self.flushReports()


class Agent:
    def __init__(self, id, env, handler):
        self.id = id
        self.env = env
        self.memory = None
        self.report = EnvReport()
        self.handler = handler
        self.player = self.env.getPlayer(self.id)

    def step(self, isTraining=True) -> None:
        if not self.env.isDone() and self.player.canStep():
            state = self.player.getState()
            action = self.handler.getAction(self.player, isTraining)
            nextState, reward, done = self.player.step(action.index)
            # print(reward)
            if self.memory is not None:
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    nextState=nextState,
                    done=done)
                self.memory.append(transition)
            # action reward
            self.report.rewards += reward

    def done(self):
        report = self.report

        # game episode reward
        doneReward = self.player.getDoneReward()
        # set last memory to done, as we may not be the last one to take action.
        # do nothing if last memory has been processed.
        if self.memory is not None and len(self.memory) > 0:
            lastMemory = self.memory[-1]
            lastMemory.done = True
            lastMemory.reward += doneReward
        report.rewards += doneReward

        # reset env variables
        self.report = EnvReport()
        return report

    def resetMemory(self, num):
        self.memory = collections.deque(maxlen=num)


class RL:
    def __init__(self, algo: Algo, gameFactory: GameFactory):
        self.algo = algo
        self.gameFactory = gameFactory

        self.lastPrint: float = 0

        mp.set_start_method("spawn")
        self.sync = SyncContext()

    def epochManagerRestartHandler(self):
        self.update(0, "\n")
        # pass

    def epochManagerTrainedHandler(self):
        self.update(0)

    def epochManagerAddHandler(self):
        self.update(0)

    async def run(self, train: bool = True, load: bool = False, episodes: int = 1000, delay: float = 0) -> None:
        self.delay = delay
        self.isTraining = train
        self.lastSave = time.perf_counter()
        self.workingPath = os.path.dirname(__main__.__file__)
        # multiprocessing.connection.BUFSIZE = 2 ** 24

        print(f"Train: {self.isTraining}")
        if self.isTraining:
            trainer = TrainerProcess(
                self.algo, self.gameFactory, self.sync, episodes, load).start()
        else:
            await Evaluator(self.algo, self.gameFactory, self.sync).eval()

        self.sync.epochManager.on("restart", self.epochManagerRestartHandler)
        # self.sync.epochManager.on("trained", self.epochManagerTrainedHandler)
        # self.sync.epochManager.on("add", self.epochManagerAddHandler)

        while True:
            self.update()
            await asyncio.sleep(0.01)

    def update(self, freq=.1, end="\b\r") -> None:
        if time.perf_counter() - self.lastPrint < freq:
            return
        epoch = self.sync.epochManager.epoch
        if epoch is not None:
            print(f"#{self.sync.epochManager.num} {Function.humanize(epoch.episodes):>6} {epoch.hitRate:>7.2%} | " +
                  f'Loss: {Function.humanize(epoch.loss):>6}/ep | ' +
                  f'Env: {Function.humanize(epoch.envs):>6} | ' +
                  f'Best: {Function.humanize(epoch.bestRewards):>6}, Avg: {Function.humanize(epoch.avgRewards):>6} | ' +
                  f'Steps: {Function.humanize(epoch.steps / epoch.duration):>6}/s | Episodes: {1 / epoch.durationPerEpisode:>6.2f}/s | ' +
                  f' {Function.humanizeTime(epoch.duration):>6} > {Function.humanizeTime(epoch.estimateDuration):}' +
                  '      ',
                  end=end)
            self.lastPrint = time.perf_counter()

