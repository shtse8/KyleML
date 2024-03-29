import __main__
import asyncio
import collections
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from enum import Enum
from multiprocessing.connection import Pipe
from multiprocessing.managers import NamespaceProxy, SyncManager
from pathlib import Path
from typing import List, Callable, TypeVar, Generic, Tuple, Any

import utils.Function as Function
from games.Game import Game
from games.GameManager import GameManager
from memories.Transition import Transition
from utils.KyleList import KyleList
from utils.Message import NetworkInfo, LearnReport, EnvReport, MethodCallRequest, MethodCallResult
from utils.Network import Network
from utils.Normalizer import RangeNormalizer, StdNormalizer
from utils.PipedProcess import Process, PipedProcess
from utils.multiprocessing import Proxy


class TensorWrapper:
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def asTensor(self):
        return self.tensor

    def asArray(self):
        return self.tensor.cpu().detach().numpy()

    def asItem(self):
        return self.tensor.item()

    def __iter__(self):
        return self.tensor.__iter__()

class Action:
    def __init__(self, index=None, probs=None, value=None):
        self.index = index
        self.probs = probs
        self.value = value

    def __int__(self):
        return self.index

    @property
    def log(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py#L72
        eps = torch.finfo(torch.float).eps
        prob = min(1-eps, max(eps, self.probs[self.index]))
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

    async def async_run(self, conn):
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

    async def async_run(self):
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
    def __init__(self, algo: Algo, gameFactory: GameManager, sync):
        super().__init__(algo, gameFactory, sync)
        self.evaluators: List[EvaluatorService] = []
        self.handler = None

    def learn(self, experience):
        steps = len(experience)
        losses = self.handler.learn(experience)
        for loss in losses:
            # learn report handling
            self.sync.epochManager.trained(loss, steps)
            self.sync.totalEpisodes.value += 1
            self.sync.totalSteps.value += steps

    def pushNewNetwork(self):
        data = self.handler.dump()
        self.sync.latestStateDict.update(data)

    async def start(self, episodes=1000, load=False):
        env = self.gameFactory.create()

        # Create Evaluators
        evaluators = []
        n_workers = max(torch.cuda.device_count() * 4, 1)
        for _ in range(n_workers):
            evaluator = EvaluatorService(self.algo, self.gameFactory, self.sync).start()
            evaluators.append(evaluator)

        self.evaluators = np.array(evaluators)

        self.handler = self.algo.createHandler(env, Role.Trainer, self.sync.getDevice())
        if load:
            self.load()
        n_samples = self.algo.config.sampleSize
        evaulator_samples = math.ceil(n_samples / n_workers)

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

class AlgoHandler:
    def __init__(self, config, env, role, device):
        self.config = config
        self.env = env
        self.device = device
        self.role = role

    def _getStateDict(self, network):
        stateDict = network.state_dict()
        for key, value in stateDict.items():
            stateDict[key] = value.cpu()  # .detach().numpy()
        return stateDict

    def dump(self):
        raise NotImplementedError

    def load(self, data):
        raise NotImplementedError

    def reset(self):
        pass

    def reportStep(self, action):
        pass

class EvaulationTerminated(Exception):
    def __init__(self):
        super().__init__("Evaulation Terminiated.")

class Evaluator(Base):
    def __init__(self, algo: Algo, gameFactory, sync):
        super().__init__(algo, gameFactory, sync)
        self.env = gameFactory.create()
        # self.algo.device = torch.device("cpu")
        # self.algo.device = sync.getDevice()
        self.handler = self.algo.createHandler(self.env, Role.Evaluator, sync.getDevice())

        # self.network = self.algo.createNetwork(
        #     self.env.observationShape, self.env.actionSpace).to(self.algo.device)
        # self.network.version = -1
        # self.networks.append(self.network)

        # self.mcts = MCTS(lambda s, m, x: self.algo.getProb(self.network, s, m, x), n_playout=100)
        self.agents = np.array([Agent(id, self.env, self.handler) for id in self.env.players])
        self.started = False

        self.reports: List[EnvReport] = []
        self.i = 0

    def updateNetwork(self):
        self.handler.load(self.sync.latestStateDict)

    def next(self, num=0):
        # auto reset
        if not self.started:
            self.env.reset()
            self.started = True
        elif self.env.is_done():
            # reports = []
            for agent in self.agents:
                self.reports.append(agent.done())
            self.env.reset()
            self.handler.reset()
            # return False

        # return True
        # memoryCount = min([len(x.memory) for x in self.agents])
        if num > 0:
            memoryCount = np.sum([len(x.memory) for x in self.agents])
            if memoryCount >= num:
                raise EvaulationTerminated()
        
        agent = self.agents[self.i]
        self.i = (self.i + 1) % len(self.agents)
        return agent

    def flushReports(self):
        if len(self.reports) > 0:
            self.sync.epochManager.add(self.reports)
            self.reports = []

    def roll(self, num):
        self.updateNetwork()
        self.generateTransitions(num)
        return [x.memory for x in self.agents]

    def generateTransitions(self, num):
        for agent in self.agents:
            agent.resetMemory(num)
        while True:
            try:
                self.next(num).step(True)
            except EvaulationTerminated:
                break
        self.flushReports()

    async def eval(self, delay=0):
        # self.agents[0] = HumanAgent(1, self.env, self.handler)
        self.env.render()
        self.load()
        self.sync.epochManager.start(1)
        while True:
            try:
                self.next(0).step(False)
                # os.system('cls')
                # state = self.env.getState(1).astype(int)
                # print(state[0] + state[1] * 2, "\n")
                await asyncio.sleep(delay)
                if self.env.is_done():
                    print("Done:")
                    for agent in self.agents:
                        print(agent.id, self.env.get_done_reward(agent.id))
                    await asyncio.sleep(3)
            except EvaulationTerminated:
                break
            self.flushReports()
     
class Agent:
    def __init__(self, id, env, handler):
        self.id = id
        self.env = env.set_player(self.id)
        self.memory = None
        self.report = EnvReport()
        self.handler = handler.getAgentHandler(self.env)

    def step(self, isTraining=True) -> None:
        if not self.env.is_done() and self.env.can_step():
            transition = Transition()
            transition.info = self.handler.get_info()
            action = self.handler.getAction(isTraining)
            reward = self.env.step(action.index)
            # print(reward)
            if self.memory is not None:
                transition.action = action
                transition.reward = reward
                transition.next = self.handler.get_info()
                self.memory.append(transition)
            # action reward
            self.report.rewards += reward

    def done(self):
        report = self.report

        # game episode reward
        doneReward = self.env.get_done_reward()
        # set last memory to done, as we may not be the last one to take action.
        # do nothing if last memory has been processed.
        if self.memory is not None and len(self.memory) > 0:
            lastMemory = self.memory[-1]
            lastMemory.next.done = True
            lastMemory.reward += doneReward
        report.rewards += doneReward

        # reset env variables
        self.report = EnvReport()
        self.handler.reset()
        return report

    def resetMemory(self, num):
        self.memory = collections.deque(maxlen=num)


class HumanAgent(Agent):
    def __init__(self, id, env, handler):
        super().__init__(id, env, handler)

    def step(self, isTraining=False):
        if not self.env.is_done() and self.env.can_step():
            while True:
                try:
                    # row, col = map(int, input(
                    #     "Your action: ").split())
                    # pos = row * self.env.size + col
                    pos = int(input("Your action: "))
                    self.env.step(pos)
                    self.handler.reportStep(pos)
                    break
                except Exception as e:
                    print(e)
               
class RL:
    def __init__(self, algo: Algo, gameFactory: GameManager):
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
            await Evaluator(self.algo, self.gameFactory, self.sync).eval(
                delay=self.delay
            )

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
                  f'Loss: {epoch.loss:.4f}/ep | ' +
                  f'Env: {Function.humanize(epoch.envs):>6} | ' +
                  f'Best: {Function.humanize(epoch.bestRewards):>6}, Avg: {Function.humanize(epoch.avgRewards):>6} | ' +
                  f'Steps: {Function.humanize(epoch.steps / epoch.duration):>6}/s | Episodes: {1 / epoch.durationPerEpisode:>6.2f}/s | ' +
                  f' {Function.humanize_time(epoch.duration):>6} > {Function.humanize_time(epoch.estimateDuration):}' +
                  '      ',
                  end=end)
            self.lastPrint = time.perf_counter()

