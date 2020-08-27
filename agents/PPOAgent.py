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
import collections
import numpy as np
from enum import Enum
import time
import sys
import multiprocessing.connection
multiprocessing.connection.BUFSIZE = 2 ** 24
# print(multiprocessing.connection.BUFSIZE)


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
    def __init__(self):
        self.index = None
        self.prediction = None

    def __int__(self):
        return self.index


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
        self.history = collections.deque(maxlen=target_episodes)
        self.bestRewards = 0

        # for stats
        self.totalRewards = 0

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
        return self.totalRewards / len(self.history) if len(self.history) > 0 else math.nan

    def add(self, report: EnvReport):
        if report.rewards > self.bestRewards:
            self.bestRewards = report.rewards
        self.totalRewards += report.rewards
        self.history.append(report)
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
    def __init__(self, inputShape, n_outputs):
        super().__init__()
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
                stateDict[key] = value.cpu().detach().numpy()
            self.info = NetworkInfo(stateDict, self.version)

    def getInfo(self) -> NetworkInfo:
        self._updateStateDict()
        return self.info

    def loadInfo(self, info: NetworkInfo):
        stateDict = info.stateDict
        for key, value in stateDict.items():
            stateDict[key] = torch.from_numpy(value)
        self.load_state_dict(stateDict)
        self.version = info.version

    def isNewer(self, info: NetworkInfo):
        return info.version > self.version


class PPONetwork(Network):
    def __init__(self, inputShape, n_outputs):
        super().__init__(inputShape, n_outputs)

        hidden_nodes = 64
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
    def __init__(self, policy: Policy):
        self.policy = policy
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def createNetwork(self) -> Network:
        raise NotImplementedError

    def getAction(self, network, state, isTraining: bool) -> PredictedAction:
        raise NotImplementedError

    def learn(self, network: Network, memory):
        raise NotImplementedError


class PPOAlgo(Algo):
    def __init__(self):
        super().__init__(Policy(
            batchSize=32,
            learningRate=3e-4,
            versionTolerance=9,
            networkUpdateStrategy=NetworkUpdateStrategy.Aggressive))
        self.gamma = 0.99
        self.epsClip = 0.2

    def createNetwork(self, inputShape, n_outputs) -> Network:
        return PPONetwork(inputShape, n_outputs).to(self.device)

    def getAction(self, network, state, isTraining: bool) -> PredictedAction:
        action = PredictedAction()
        network.eval()
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float, device=self.device)
            prediction = network.getPolicy(state).squeeze(0)
            action.prediction = prediction.cpu().detach().numpy()
            if isTraining:
                action.index = torch.distributions.Categorical(
                    probs=prediction).sample().item()
            else:
                action.index = prediction.argmax().item()
            return action

    def getGAE(self, rewards, dones, values, lastValue=0):
        advantages = np.zeros_like(rewards).astype(float)
        gae = 0
        for i in reversed(range(len(rewards))):
            detlas = rewards[i] + self.gamma * \
                lastValue * (1 - dones[i]) - values[i]
            gae = detlas + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages[i] = gae
            lastValue = values[i]
        return advantages

    def learn(self, network: Network, memory):
        network.train()

        states = np.array([x.state for x in memory])        
        states = torch.tensor(states, dtype=torch.float, device=self.device)

        actions = np.array([x.action.index for x in memory])
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        predictions = np.array([x.action.prediction for x in memory])
        predictions = torch.tensor(predictions, dtype=torch.float, device=self.device)

        dones = np.array([x.done for x in memory])
        rewards = np.array([x.reward for x in memory])
        old_log_probs = torch.distributions.Categorical(
            probs=predictions).log_prob(actions)

        
        lastValue = 0
        if not dones[-1]:
            lastState = torch.tensor([memory[-1].nextState], dtype=torch.float, device=self.device)
            lastValue = network.getValue(lastState).item()
        # returns = self.getDiscountedRewards(rewards, dones, lastValue)
        # returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        # returns = Function.normalize(returns)
        
        action_probs, values = network(states)
        values = values.squeeze(1)

        # GAE (General Advantage Estimation)
        # Paper: https://arxiv.org/abs/1506.02438
        # Code: https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L55-L64
        advantages = self.getGAE(
            rewards, dones, values.cpu().detach().numpy(), lastValue)
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)

        # from baseline
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L65
        returns = advantages + values.detach()

        # Normalize advantages
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L139
        advantages = Function.normalize(advantages)

        dist = torch.distributions.Categorical(probs=action_probs)

        # porb1 / porb2 = exp(log(prob1) - log(prob2))
        ratios = torch.exp(dist.log_prob(actions) - old_log_probs)
        policy_losses1 = ratios * advantages
        policy_losses2 = ratios.clamp(1 - self.epsClip, 1 + self.epsClip) * advantages

        # Maximize Policy Loss (Rewards)
        policy_loss = -torch.min(policy_losses1, policy_losses2).mean()

        # Maximize Entropy Loss
        entropy_loss = -dist.entropy().mean()  
        
        # Minimize Value Loss  (MSE)
        value_loss = (returns - values).pow(2).mean()

        loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss

        tic = time.perf_counter()
        network.optimizer.zero_grad()
        loss.backward()

        # Chip grad with norm
        # https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/ppo2/model.py#L107
        nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
        network.optimizer.step()

        # Report
        network.version += 1
        loss_float = loss.item()

        return loss_float


class Base:
    def __init__(self, algo: Algo, env):
        self.algo: Algo = algo
        self.env = env


class Trainer(Base):
    def __init__(self, algo: Algo, env, conn):
        super().__init__(algo, env)
        self.network = self.algo.createNetwork(
            self.env.observationShape, self.env.actionSpace).buildOptimizer(self.algo.policy.learningRate)
        self.conn = conn
        self.lastBroadcast = None
        self.memoryPushBuffer = collections.deque(maxlen=1000)

    def recv(self):
        while self.conn.poll():
            message = self.conn.recv()
            if isinstance(message, MemoryPush):
                self.memoryPushBuffer.append(message)

    def learn(self):
        while len(self.memoryPushBuffer):
            message = self.memoryPushBuffer.popleft()
            steps = len(message.memory)
            if message.version >= self.network.version - self.algo.policy.versionTolerance:
                loss = self.algo.learn(self.network, message.memory)
                self.conn.send(LearnReport(loss=loss, steps=steps))
            else:
                self.conn.send(LearnReport(drops=steps))

    def pushNewNetwork(self):
        if self.lastBroadcast is None or self.lastBroadcast.version < self.network.version:
            networkInfo = self.network.getInfo()
            self.conn.send(networkInfo)
            self.lastBroadcast = networkInfo

    def start(self, isTraining=False):
        while True:
            self.recv()
            self.learn()
            self.pushNewNetwork()


class Evaluator(Base):
    def __init__(self, id, algo: Algo, env, conn, delay=0):
        super().__init__(algo, env.getNew())
        self.id = id
        self.delay = delay
        # self.algo.device = torch.device("cpu")
        self.network = self.algo.createNetwork(
            self.env.observationShape, self.env.actionSpace)
        self.network.version = -1
        self.conn = conn

        self.memory = collections.deque(maxlen=self.algo.policy.batchSize)
        self.report = None

        self.lastestNetworkInfo = None
        self.waiting = True

    def recv(self):
        while self.conn.poll():
            message = self.conn.recv()
            if isinstance(message, NetworkInfo):
                self.lastestNetworkInfo = message
                self.waiting = False

    def applyNextNetwork(self):
        if self.lastestNetworkInfo and self.network.isNewer(self.lastestNetworkInfo):
            self.network.loadInfo(self.lastestNetworkInfo)
            self.memory.clear()
            # print("Applied new network", self.network.version)
            return True
        return False

    def pushMemory(self):
        if self.isValidVersion():
            self.conn.send(MemoryPush(self.memory, self.network.version))
        self.memory.clear()
        self.waiting = True
        # time.sleep(0.1)

    def commit(self, transition: Transition):
        self.report.rewards += transition.reward
        self.memory.append(transition)
        if len(self.memory) >= self.algo.policy.batchSize:
            self.pushMemory()
            # self.applyNextNetwork()

    def isValidVersion(self):
        return self.lastestNetworkInfo and self.network.version >= self.lastestNetworkInfo.version - self.algo.policy.versionTolerance

    def checkVersion(self):
        if self.algo.policy.networkUpdateStrategy == NetworkUpdateStrategy.Aggressive or not self.isValidVersion():
            self.applyNextNetwork()
        return self.isValidVersion()

    def start(self, isTraining=False):
        while True:
            self.report = EnvReport()
            state = self.env.reset()
            done: bool = False
            while not done:
                self.recv()
                if self.waiting or not self.checkVersion():
                    time.sleep(0.01)
                    continue

                action = self.algo.getAction(self.network, state, isTraining)
                nextState, reward, done = self.env.takeAction(action.index)
                transition = Transition(state, action, reward, nextState, done)
                self.commit(transition)
                if self.delay > 0:
                    time.sleep(self.delay)
                state = nextState
            self.conn.send(self.report)


class Agent:
    def __init__(self, algo: Algo, env):
        self.evaluators = []
        self.algo = algo
        self.env = env
        self.history = []

        self.totalEpisodes = 0
        self.totalSteps = 0
        self.epochs = 1
        self.dropped = 0
        self.lastPrint = 0

    def broadcast(self, message):
        for evaluator in self.evaluators:
            evaluator.send(message)

    def run(self, train: bool = True, episodes: int = 10000, delay: float = 0) -> None:
        self.delay = delay
        self.isTraining = train
        # mp.set_start_method("spawn")

        # Create Evaluators
        print(
            f"Train: {self.isTraining}, Total Episodes: {self.totalEpisodes}, Total Steps: {self.totalSteps}")

        evaluators = []
        # n_workers = mp.cpu_count() - 1
        # n_workers = mp.cpu_count() // 2
        n_workers = 2
        for i in range(n_workers):
            evaluator = EvaluatorProcess(
                i, self.algo, self.env, self.isTraining).start()
            evaluators.append(evaluator)

        self.evaluators = np.array(evaluators)
        trainer = TrainerProcess(self.algo, self.env).start()
        self.epoch = Epoch(episodes).start()
        while True:
            self.update()

            # print("Evaluators Poll")
            for evaluator in self.evaluators:
                if evaluator.poll():
                    message = evaluator.recv()
                    if isinstance(message, MemoryPush):
                        trainer.send(message)
                    elif isinstance(message, EnvReport):
                        self.epoch.add(message)
                    else:
                        raise Exception("Unknown Message")

            if trainer.poll():
                message = trainer.recv()
                if isinstance(message, NetworkInfo):
                    self.broadcast(message)
                elif isinstance(message, LearnReport):
                    if message.steps > 0:
                        self.epoch.trained(message.loss, message.steps)
                        self.totalSteps += message.steps
                        if self.epoch.isEnd:
                            self.update(0)
                            print()
                            self.epoch = Epoch(episodes).start()
                            self.epochs += 1
                    else:
                        self.epoch.drops += message.drops

    def update(self, freq=.1) -> None:
        if time.perf_counter() - self.lastPrint < freq:
            return
        print(f"#{self.epochs} {Function.humanize(self.epoch.episodes):>6} {self.epoch.hitRate:>7.2%} | " +
              f'Loss: {Function.humanize(self.epoch.loss):>6}/ep | ' +
              f'Env: {Function.humanize(len(self.epoch.history)):>6} | ' +
              f'Best: {Function.humanize(self.epoch.bestRewards):>6}, Avg: {Function.humanize(self.epoch.avgRewards):>6} | ' +
              f'Steps: {Function.humanize(self.epoch.steps / self.epoch.duration):>5}/s | Episodes: {1 / self.epoch.durationPerEpisode:>6.2f}/s | ' +
              f' {Function.humanizeTime(self.epoch.duration):>5} > {Function.humanizeTime(self.epoch.estimateDuration):}' +
              '      ',
              end="\b\r")
        self.lastPrint = time.perf_counter()


class PipedProcess:
    def __init__(self):
        self.process = None
        self.conn = None
        self.started = False

    def start(self):
        if self.started:
            raise Exception("Process is started")

        self.started = True
        self.conn, child_conn = mp.Pipe(True)
        self.process = mp.Process(target=self.run, args=(child_conn,))
        self.process.start()
        return self

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, obj):
        return self.conn.send(obj)

    def run(self, conn):
        pass


class EvaluatorProcess(PipedProcess):
    def __init__(self, id, algo, env, isTraining):
        super().__init__()
        self.id = id
        self.algo = algo
        self.env = env
        self.isTraining = isTraining

    def run(self, conn):
        Evaluator(self.id, self.algo, self.env, conn).start(self.isTraining)


class TrainerProcess(PipedProcess):
    def __init__(self, algo, env):
        super().__init__()
        self.algo = algo
        self.env = env

    def run(self, conn):
        Trainer(self.algo, self.env, conn).start()
