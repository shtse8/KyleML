import asyncio
import __main__
from pathlib import Path
import os
import math
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as schedular
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import utils.Function as Function
from utils.PredictionHandler import PredictionHandler
from .Agent import Agent
from memories.Transition import Transition
from memories.SimpleMemory import SimpleMemory
from games.GameFactory import GameFactory
import collections
import numpy as np
from enum import Enum
import time
import sys
import multiprocessing.connection
from utils.PipedProcess import Process, PipedProcess


# def init_layer(m):
#     weight = m.weight.data
#     weight.normal_(0, 1)
#     weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
#     nn.init.constant_(m.bias.data, 0)
#     return m


class Message:
    def __init__(self):
        pass


class NetworkUpdateStrategy(Enum):
    Aggressive = 1
    Lazy = 2


class NetworkInfo(Message):
    def __init__(self, stateDict, version):
        self.stateDict = stateDict
        self.version = version


class MemoryPush(Message):
    def __init__(self, memory, version):
        super().__init__()
        self.memory = memory
        self.version = version


class LearnReport(Message):
    def __init__(self, loss=0, steps=0, drops=0):
        self.loss = loss
        self.steps = steps
        self.drops = drops


class EnvReport(Message):
    def __init__(self):
        self.rewards = 0


class PredictedAction(object):
    def __init__(self, index, prediction):
        self.index = index
        self.prediction = prediction

    def __int__(self):
        return self.index

    @property
    def log(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py#L72
        eps = torch.finfo(torch.float).eps
        prob = min(1-eps, max(eps, self.prediction[self.index]))
        return math.log(prob)


class Epoch(Message):
    def __init__(self, target_episodes):
        self.target_episodes = target_episodes
        self.steps: int = 0
        self.drops: int = 0
        self.rewards: float = 0
        self.total_loss: float = 0
        self.epoch_start_time: int = 0
        self.epoch_end_time: int = 0
        self.episodes = 0

        # for stats
        # self.history = collections.deque(maxlen=target_episodes)
        self.bestRewards = -math.inf
        self.totalRewards = 0
        self.envs = 0

    def start(self):
        self.epoch_start_time = time.perf_counter()
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
        return self.total_loss / self.steps if self.steps > 0 else 0

    @property
    def durationPerEpisode(self):
        return self.duration / self.episodes if self.episodes > 0 else math.inf

    @property
    def estimateDuration(self):
        return self.target_episodes * self.durationPerEpisode

    @property
    def avgRewards(self):
        return self.totalRewards / self.envs if self.envs > 0 else math.nan

    def add(self, report: EnvReport):
        if report.rewards > self.bestRewards:
            self.bestRewards = report.rewards
        self.totalRewards += report.rewards
        self.envs += 1
        # self.history.append(report)
        return self

    def trained(self, loss, steps):
        self.total_loss += loss * steps
        self.steps += steps
        self.episodes += 1
        if self.episodes >= self.target_episodes:
            self.end()
        return self


class ConvLayers(nn.Module):
    def __init__(self, inputShape, n_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.MaxPool2d(1),
            nn.Flatten(),
            nn.Linear(64 * inputShape[1] * inputShape[2], n_outputs),
            nn.ReLU())

    def forward(self, x):
        return self.layers(x)




class FCLayers(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_nodes=0):
        super().__init__()
        if hidden_nodes == 0:
            hidden_nodes = n_outputs

        self.layers = nn.Sequential(
                nn.Linear(n_inputs, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, n_outputs),
                nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class BodyLayers(nn.Module):
    def __init__(self, inputShape, n_outputs):
        super().__init__()
        if type(inputShape) is tuple and len(inputShape) == 3:
            self.layers = ConvLayers(inputShape, n_outputs)
        else:
            if type(inputShape) is tuple and len(inputShape) == 1:
                inputShape = inputShape[0]
            self.layers = FCLayers(inputShape, n_outputs)

    def forward(self, x):
        return self.layers(x)


class Network(nn.Module):
    def __init__(self, inputShape, n_outputs, name="network"):
        super().__init__()
        self.name = name
        self.optimizer = None
        self.version: int = 1
        self.info: NetworkInfo = None

    def buildOptimizer(self):
        raise NotImplementedError

    def _updateStateDict(self):
        if self.info is None or self.info.version != self.version:
            # print("Update Cache", self.version)
            stateDict = self.state_dict()
            for key, value in stateDict.items():
                stateDict[key] = value.cpu() #.detach().numpy()
            self.info = NetworkInfo(stateDict, self.version)

    def getInfo(self) -> NetworkInfo:
        self._updateStateDict()
        return self.info

    def loadInfo(self, info: NetworkInfo):
        stateDict = info.stateDict
        # for key, value in stateDict.items():
        #     stateDict[key] = torch.from_numpy(value)
        self.load_state_dict(stateDict)
        self.version = info.version

    def isNewer(self, info: NetworkInfo):
        return info.version > self.version


class PPONetwork(Network):
    def __init__(self, inputShape, n_outputs):
        super().__init__(inputShape, n_outputs)

        hidden_nodes = 512
        self.body = BodyLayers(inputShape, hidden_nodes)

        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, n_outputs),
            nn.Softmax(dim=-1))

        # Define value head
        self.value = nn.Linear(hidden_nodes, 1)

    def buildOptimizer(self, learningRate):
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate, eps=1e-5)
        return self

    def forward(self, state):
        output = self.body(state)
        return self.policy(output), self.value(output)

    def getPolicy(self, state):
        output = self.body(state)
        return self.policy(output)

    def getValue(self, state):
        output = self.body(state)
        return self.value(output)


class Policy:
    def __init__(self, batchSize, learningRate, versionTolerance, networkUpdateStrategy):
        self.batchSize = batchSize
        self.versionTolerance = versionTolerance
        self.learningRate = learningRate
        self.networkUpdateStrategy = networkUpdateStrategy


class OnPolicy(Policy):
    def __init__(self, batchSize, learningRate):
        super().__init__(batchSize, learningRate, 0, NetworkUpdateStrategy.Aggressive)


class OffPolicy(Policy):
    def __init__(self, batchSize, learningRate):
        super().__init__(batchSize, learningRate, math.inf, NetworkUpdateStrategy.Lazy)


class Algo:
    def __init__(self, name, policy: Policy):
        self.name = name
        self.policy = policy
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def createNetwork(self) -> Network:
        raise NotImplementedError

    def getAction(self, network, state, mask, isTraining: bool) -> PredictedAction:
        raise NotImplementedError

    def learn(self, network: Network, memory):
        raise NotImplementedError


class PPOAlgo(Algo):
    def __init__(self):
        super().__init__("PPO", Policy(
            batchSize=512,
            learningRate=3e-4,
            versionTolerance=9,
            networkUpdateStrategy=NetworkUpdateStrategy.Aggressive))
        self.gamma = 0.99
        self.epsClip = 0.2
        self.gaeCoeff = 0.95

    def createNetwork(self, inputShape, n_outputs) -> Network:
        return PPONetwork(inputShape, n_outputs)

    def getAction(self, network, state, prediction, mask, isTraining: bool) -> PredictedAction:
        network.eval()
        with torch.no_grad():
            if prediction is None:
                state = torch.tensor([state], dtype=torch.float, device=self.device)
                prediction = network.getPolicy(state).squeeze(0)
                prediction = prediction.cpu().detach().numpy()
            # print(prediction, mask)
            handler = PredictionHandler(prediction.copy(), mask)
            # print(prediction, mask)
            index = handler.getRandomAction()
            # print(index, prediction)
            return PredictedAction(
                index=index,
                prediction=prediction
            )
            # if isTraining:
            #     index = torch.distributions.Categorical(
            #         probs=prediction).sample().item()
            # else:
            #     index = prediction.argmax().item()
            # return PredictedAction(
            #     index=index,
            #     prediction=prediction.cpu().detach().numpy()
            # )

    def processAdvantage(self, network, memory):
        with torch.no_grad():
            lastValue = 0
            lastMemory = memory[-1]
            if not lastMemory.done:
                lastState = torch.tensor([lastMemory.nextState], dtype=torch.float, device=self.device)
                lastValue = network.getValue(lastState).item()

            states = np.array([x.state for x in memory])
            states = torch.tensor(states, dtype=torch.float, device=self.device)

            values = network.getValue(states).squeeze(1).cpu().detach().numpy()

            # GAE (General Advantage Estimation)
            # Paper: https://arxiv.org/abs/1506.02438
            # Code: https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L55-L64
            gae = 0
            for i in reversed(range(len(memory))):
                transition = memory[i]
                detlas = transition.reward + self.gamma * lastValue * (1 - transition.done) - values[i]
                gae = detlas + self.gamma * self.gaeCoeff * gae * (1 - transition.done)
                # from baseline
                # https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L65
                transition.advantage = gae
                transition.reward = gae + values[i]
                transition.value = values[i]
                lastValue = values[i]

            
            # Normalize advantages
            # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L139
            advantages = np.array([x.advantage for x in memory])
            advantages = Function.normalize(advantages)
            for transition, advantage in zip(memory, advantages):
                transition.advantage = advantage

    def getGAE(self, rewards, dones, values, lastValue=0):
        advantages = np.zeros_like(rewards).astype(float)
        gae = 0
        for i in reversed(range(len(rewards))):
            detlas = rewards[i] + self.gamma * \
                lastValue * (1 - dones[i]) - values[i]
            gae = detlas + self.gamma * self.gaeCoeff * gae * (1 - dones[i])
            advantages[i] = gae
            lastValue = values[i]
        return advantages

    def learn(self, network: Network, memory):
        network.train()
        memory = np.array(memory)
        miniBatchSize = 32
        n_miniBatch = len(memory) // miniBatchSize
        totalLosses = 0
        totalSamples = 0
        network.optimizer.zero_grad()
        for i in range(n_miniBatch):
            startIndex = i * miniBatchSize
            endIndex = startIndex + miniBatchSize
            minibatch = memory[startIndex:endIndex]
            
            states = np.array([x.state for x in minibatch])
            states = torch.tensor(states, dtype=torch.float, device=self.device).detach()

            actions = np.array([x.action.index for x in minibatch])
            actions = torch.tensor(actions, dtype=torch.long, device=self.device).detach()

            old_log_probs = np.array([x.action.log for x in minibatch])
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float, device=self.device).detach()

            returns = np.array([x.reward for x in minibatch])
            returns = torch.tensor(returns, dtype=torch.float, device=self.device).detach()

            old_values = np.array([x.value for x in minibatch])
            old_values = torch.tensor(old_values, dtype=torch.float, device=self.device).detach()

            # advantages = returns - old_values
            advantages = np.array([x.advantage for x in minibatch])
            advantages = torch.tensor(advantages, dtype=torch.float, device=self.device).detach()

            action_probs, values = network(states)
            values = values.squeeze(1)

            # PPO2 - Confirm the samples aren't too far from pi.
            # porb1 / porb2 = exp(log(prob1) - log(prob2))
            dist = torch.distributions.Categorical(probs=action_probs)
            ratios = torch.exp(dist.log_prob(actions) - old_log_probs)
            policy_losses1 = ratios * advantages
            policy_losses2 = ratios.clamp(1 - self.epsClip, 1 + self.epsClip) * advantages

            # Maximize Policy Loss (Rewards)
            policy_loss = -torch.min(policy_losses1, policy_losses2).mean()

            # Maximize Entropy Loss
            entropy_loss = -dist.entropy().mean()
            
            # Minimize Value Loss  (MSE)
            # Clip the value to reduce variability during Critic training
            # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L66-L75
            value_loss = (returns - values).pow(2).mean()
            #value_loss1 = (returns - values).pow(2)
            #valuesClipped = old_values + torch.clamp(values - old_values, -self.epsClip, self.epsClip)
            #value_loss2 = (returns - valuesClipped).pow(2)
            #value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            #print(value_loss1, valuesClipped)

            # Calculating Total loss
            # Wondering  if we need to divide the number of minibatches to keep the same learning rate?
            # As the learning rate is a parameter of optimizer, and only one step is called. 
            # Should be fine to not dividing the number of minibatches.
            loss = (policy_loss + 0.01 * entropy_loss + 0.5 * value_loss) / n_miniBatch

            # losses.append(loss)
            # Accumulating the loss to the graph
            loss.backward()
            totalLosses += loss.item() * len(minibatch)
            totalSamples += len(minibatch)
        # print(torch.cat(losses), loss)

        # loss = torch.cat(losses).mean()
        # loss.backward()

        # Chip grad with norm
        # https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/ppo2/model.py#L107
        nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
  
        network.optimizer.step()
        network.version += 1

        return totalLosses / totalSamples


class Base:
    def __init__(self, algo: Algo):
        self.algo: Algo = algo


class Trainer(Base):
    def __init__(self, network, algo: Algo, gameFactory: GameFactory, sync):
        super().__init__(algo)
        self.gameFactory = gameFactory
        self.algo.device = sync.getDevice()
        self.network = network
        self.lastBroadcast = None
        self.sync = sync
        self.evaluators = []

    def learn(self, memory):
        steps = len(memory)
        loss = self.algo.learn(self.network, memory)
        report = LearnReport(loss=loss, steps=steps)
        self.sync.reportQueue.put(report)

    def pushNewNetwork(self):
        if self.lastBroadcast is None or self.lastBroadcast.version < self.network.version:
            networkInfo = self.network.getInfo()
            self.sync.latestStateDict.update(networkInfo.stateDict)
            self.sync.latestVersion.value = networkInfo.version
            self.lastBroadcast = networkInfo

    async def start(self, isTraining=False):
        evaluators = []
        n_workers = max(torch.cuda.device_count(), 1)
        for i in range(n_workers):
            evaluator = EvaluatorService(self.network, self.algo, self.gameFactory, self.sync).start()
            evaluators.append(evaluator)

        self.evaluators = np.array(evaluators)

        self.network = self.network.buildOptimizer(self.algo.policy.learningRate).to(self.algo.device)
        n_samples = 512 * n_workers
        evaulator_samples = math.ceil(n_samples / n_workers)
        while True:
            # push new network
            self.pushNewNetwork()
            # collect samples
            memory = collections.deque(maxlen=n_samples)
            promises = np.array([x.call("roll", (evaulator_samples,)) for x in self.evaluators])
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.as_completed
            for promise in asyncio.as_completed(promises):
                response = await promise  # earliest result
                # print("Response", response.result)
                # print("Rolled Memory: ", len(response.result))
                memory.extend(response.result)
            # learn
            self.learn(memory)


class Evaluator(Base):
    def __init__(self, network, algo: Algo, gameFactory, sync):
        super().__init__(algo)
        self.gameFactory = gameFactory
        self.env = gameFactory.get()
        # self.algo.device = torch.device("cpu")
        self.algo.device = sync.getDevice()
        self.network = network.to(self.algo.device)
        self.network.version = -1
        self.sync = sync

        self.report = None
        self.generator = None

    def updateNetwork(self):
        if self.network.version < self.sync.latestVersion.value:
            networkInfo = NetworkInfo(self.sync.latestStateDict, self.sync.latestVersion.value)
            self.network.loadInfo(networkInfo)

    def roll(self, num):
        if self.generator is None:
            self.generator = self.transitionGenerator()
        self.updateNetwork()
        memory = np.array([next(self.generator) for _ in range(num)])
        self.algo.processAdvantage(self.network, memory)
        return memory
        # message = MemoryPush(memory, self.network.version)
        # self.sync.memoryQueue.put(message)

    def transitionGenerator(self):
        while True:
            self.report = EnvReport()
            state = self.env.reset()
            done: bool = False
            while not done:
                actionMask = np.ones(self.env.actionSpace, dtype=int)
                prediction = None
                while True:
                    try:
                        action = self.algo.getAction(self.network, state, prediction, actionMask, False)
                        nextState, reward, done = self.env.takeAction(action.index)
                        break
                    except Exception:
                        actionMask[action.index] = 0
                        prediction = action.prediction
                # print(state, action, reward, nextState, done)
                transition = Transition(state, action, reward, nextState, done)
                self.report.rewards += transition.reward
                yield transition
                state = nextState
            self.sync.reportQueue.put(self.report)

class Agent:
    def __init__(self, algo: Algo, gameFactory: GameFactory):
        self.evaluators = []
        self.algo = algo
        self.gameFactory = gameFactory

        self.totalEpisodes = 0
        self.totalSteps = 0
        self.epochs = 1
        self.dropped = 0
        self.lastPrint = 0
        self.weightPath = "./weights/"
        self.networks = []

        mp.set_start_method("spawn")
        self.sync = SyncContext()

    def broadcast(self, message):
        for evaluator in self.evaluators:
            evaluator.send(message)

    def run(self, train: bool = True, load: bool = False, episodes: int = 1000, delay: float = 0) -> None:
        self.delay = delay
        self.isTraining = train
        self.lastSave = time.perf_counter()
        # multiprocessing.connection.BUFSIZE = 2 ** 24

        env = self.gameFactory.get()
        network = self.algo.createNetwork(env.observationShape, env.actionSpace)
        self.networks.append(network)
        if not train or load:
            self.load()
        print(
            f"Train: {self.isTraining}, Trained: {Function.humanize(self.totalEpisodes)} episodes, {Function.humanize(self.totalSteps)} steps")

        trainer = TrainerProcess(network, self.algo, self.gameFactory, self.sync).start()
        self.epoch = Epoch(episodes).start()
        while True:
            self.update()

            while not self.sync.reportQueue.empty():
                message = self.sync.reportQueue.get()
                if isinstance(message, LearnReport):
                    if message.steps > 0:
                        self.epoch.trained(message.loss, message.steps)
                        self.totalEpisodes += 1
                        self.totalSteps += message.steps
                        if time.perf_counter() - self.lastSave > 60:
                            self.save()
                        if self.epoch.isEnd:
                            self.update(0)
                            print()
                            self.epoch = Epoch(episodes).start()
                            self.epochs += 1
                    else:
                        self.epoch.drops += message.drops
                elif isinstance(message, EnvReport):
                    self.epoch.add(message)
                    
            time.sleep(0.01)

    def update(self, freq=.1) -> None:
        if time.perf_counter() - self.lastPrint < freq:
            return
        print(f"#{self.epochs} {Function.humanize(self.epoch.episodes):>6} {self.epoch.hitRate:>7.2%} | " +
              f'Loss: {Function.humanize(self.epoch.loss):>6}/ep | ' +
              f'Env: {Function.humanize(self.epoch.envs):>6} | ' +
              f'Best: {Function.humanize(self.epoch.bestRewards):>6}, Avg: {Function.humanize(self.epoch.avgRewards):>6} | ' +
              f'Steps: {Function.humanize(self.epoch.steps / self.epoch.duration):>6}/s | Episodes: {1 / self.epoch.durationPerEpisode:>6.2f}/s | ' +
              f' {Function.humanizeTime(self.epoch.duration):>5} > {Function.humanizeTime(self.epoch.estimateDuration):}' +
              '      ',
              end="\b\r")
        self.lastPrint = time.perf_counter()

    def save(self) -> None:
        try:
            path = self.getSavePath(True)
            data = {
                "totalSteps": self.totalSteps,
                "totalEpisodes": self.totalEpisodes
            }
            for network in self.networks:
                data[network.name] = network.state_dict()
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
            self.totalSteps = int(data["totalSteps"]) if "totalSteps" in data else 0
            self.totalEpisodes = int(data["totalEpisodes"]) if "totalEpisodes" in data else 0
            for network in self.networks:
                print(f"{network.name} weights loaded.")
                network.load_state_dict(data[network.name])
        except Exception as e:
            print("Failed to load.", e)
    
    def getSavePath(self, makeDir: bool = False) -> str:
        path = os.path.join(os.path.dirname(__main__.__file__), self.weightPath, self.algo.name.lower(), self.gameFactory.name + ".h5")
        if makeDir:
            Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        return path


class MethodCallRequest(Message):
    def __init__(self, method, args):
        self.method = method
        self.args = args


class MethodCallResult(Message):
    def __init__(self, result):
        self.result = result

class Promise:
    def __init__(self):
        self.result = None

class Service(PipedProcess):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory
        self.isRunning = True

    async def asyncRun(self, conn):
        # print("Evaluator", os.getpid(), conn)
        self.object = self.factory()
        while self.isRunning:
            if conn.poll():
                message = conn.recv()
                if isinstance(message, MethodCallRequest):
                    # print("MMethodCallRequest", message.method)
                    result = getattr(self.object, message.method)(*message.args)
                    conn.send(MethodCallResult(result))

    def call(self, method, args=()):
        # print("Call", method)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.send(MethodCallRequest(method, args))

        async def waitResponse():
            while True:
                if self.poll():
                    message = self.recv()
                    future.set_result(message)
                    break
                await asyncio.sleep(0)

        loop.create_task(waitResponse())
        return future

class EvaluatorService(Service):
    def __init__(self, network, algo, gameFactory, sync):
        self.network = network
        self.algo = algo
        self.gameFactory = gameFactory
        self.sync = sync
        super().__init__(self.factory)

    def factory(self):
        return Evaluator(self.network, self.algo, self.gameFactory, self.sync)

class TrainerProcess(Process):
    def __init__(self, network, algo, gameFactory, sync):
        super().__init__()
        self.algo = algo
        self.network = network
        self.gameFactory = gameFactory
        self.sync = sync

    async def asyncRun(self):
        # print("Trainer", os.getpid())
        await Trainer(self.network, self.algo, self.gameFactory, self.sync).start()


# thread safe
class SyncContext:
    def __init__(self):
        self.latestStateDict = mp.Manager().dict()
        self.latestVersion = mp.Value('i', -1)
        self.memoryQueue = mp.Queue(maxsize=1000)
        self.reportQueue = mp.Queue(maxsize=1000)
        self.deviceIndex = mp.Value('i', 0)

    def getDevice(self):
        deviceName = "cpu"
        if torch.cuda.is_available():
            cudaId = self.deviceIndex.value % torch.cuda.device_count()
            deviceName = "cuda:" + str(cudaId)
            self.deviceIndex.value = self.deviceIndex.value + 1
        return torch.device(deviceName)
