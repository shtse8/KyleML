import sys
import time
from enum import Enum
import numpy as np
import collections
from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from .Agent import Agent
from utils.PredictionHandler import PredictionHandler
import utils.Function as Function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as schedular
import torch.multiprocessing as mp
import math
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
    def __init__(self, loss, steps, drops=0):
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
        if type(inputShape) is tuple and len(inputShape) == 3:
            self.body = nn.Sequential(
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
                nn.Linear(64 * inputShape[1] * inputShape[2], hidden_nodes),
                nn.ReLU())
        else:

            if type(inputShape) is tuple and len(inputShape) == 1:
                inputShape = inputShape[0]

            self.body = nn.Sequential(
                nn.Linear(inputShape, hidden_nodes),
                nn.ReLU(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU())

        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, n_outputs),
            nn.Softmax(dim=-1))

        # Define value head
        self.value = nn.Sequential(nn.Linear(hidden_nodes, 1))

    def buildOptimizer(self, learningRate):
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
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

    def getAction(self, network, state, mask, isTraining: bool) -> PredictedAction:
        raise NotImplementedError

    def learn(self, network: Network, memory):
        raise NotImplementedError


class PPOAlgo(Algo):
    def __init__(self):
        super().__init__(Policy(
            batchSize=32,
            learningRate=0.0001,
            versionTolerance=9,
            networkUpdateStrategy=NetworkUpdateStrategy.Aggressive))
        self.gamma = 0.99
        self.epsClip = 0.2

    def createNetwork(self, inputShape, n_outputs) -> Network:
        return PPONetwork(inputShape, n_outputs).to(self.device)

    def getAction(self, network, state, mask, isTraining: bool) -> PredictedAction:
        action = PredictedAction()
        network.eval()
        with torch.no_grad():
            state = torch.FloatTensor([state]).to(self.device)
            prediction = network.getPolicy(state).squeeze(0)
            action.prediction = prediction.cpu().detach().numpy()
            handler = PredictionHandler(action.prediction, mask)
            action.index = handler.getRandomAction() if isTraining else handler.getBestAction()

            # if isTraining:
            #     action.index = torch.distributions.Categorical(probs=prediction).sample().item()
            # else:
            #     action.index = prediction.argmax().item()
            return action

    # Discounted Rewards (N-steps)
    def getDiscountedRewards(self, rewards, dones, lastValue=0):
        discountedRewards = np.zeros_like(rewards).astype(float)
        for i in reversed(range(len(rewards))):
            lastValue = rewards[i] + self.gamma * lastValue * (1 - dones[i])
            discountedRewards[i] = lastValue
        return discountedRewards

    def getAdvantages(self, rewards, dones, values, lastValue=0):
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
        states = torch.FloatTensor(states).to(self.device)

        actions = np.array([x.action.index for x in memory])
        actions = torch.LongTensor(actions).to(self.device)

        predictions = np.array([x.action.prediction for x in memory])
        predictions = torch.FloatTensor(predictions).to(self.device).detach()
        dones = np.array([x.done for x in memory])
        rewards = np.array([x.reward for x in memory])
        real_probs = torch.distributions.Categorical(
            probs=predictions).log_prob(actions)

        lastValue = 0
        if not dones[-1]:
            nextStates = np.array([x.nextState for x in memory])
            lastState = torch.FloatTensor([nextStates[-1]]).to(self.device)
            lastValue = network.getValue(lastState).item()
        targetValues = self.getDiscountedRewards(rewards, dones, lastValue)
        targetValues = torch.FloatTensor(targetValues).to(self.device)
        # targetValues = Function.normalize(targetValues)

        action_probs, values = network(states)
        values = values.squeeze(1)

        # advantages = targetValues - values.detach()
        # print(targetValues, values)
        # print(advantages)
        advantages = self.getAdvantages(
            rewards, dones, values.cpu().detach().numpy(), lastValue)
        advantages = torch.FloatTensor(advantages).to(self.device)
        # print(advantages)
        # advantages = Function.normalize(advantages)

        dist = torch.distributions.Categorical(probs=action_probs)
        # porb1 / porb2 = exp(log(prob1) - log(prob2))
        ratios = torch.exp(dist.log_prob(actions) - real_probs)
        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.epsClip, 1 + self.epsClip) * advantages

        # Maximize Policy Loss (Rewards)
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -dist.entropy().mean()  # Maximize Entropy Loss
        # Minimize Value Loss (Distance to Target)
        value_loss = F.mse_loss(values, targetValues)
        loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss
        # print(policy_loss, entropy_loss, value_loss, loss)

        network.optimizer.zero_grad()
        loss.backward()
        # Chip grad with norm
        nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
        network.optimizer.step()

        # Report
        network.version += 1
        return np.mean(loss.item())


class Base:
    def __init__(self, algo: Algo, env):
        self.algo: Algo = algo
        self.env = env


class Trainer(Base):
    def __init__(self, algo: Algo, env):
        super().__init__(algo, env)
        self.network = self.algo.createNetwork(
            self.env.observationShape, self.env.actionSpace).buildOptimizer(self.algo.policy.learningRate)

    def learn(self, memory):
        return self.algo.learn(self.network, memory)


class Evaluator(Base):
    def __init__(self, algo: Algo, env, conn, delay=0):
        super().__init__(algo, env.getNew())
        self.delay = delay
        # self.algo.device = torch.device("cpu")
        self.network = self.algo.createNetwork(
            self.env.observationShape, self.env.actionSpace)
        self.network.version = -1
        self.conn = conn
        self.isRequesting = False
        self.memory = collections.deque(maxlen=self.algo.policy.batchSize)
        self.report = None

        self.lastestNetworkInfo = None

    def recv(self):
        while self.conn.poll():
            message = self.conn.recv()
            if isinstance(message, NetworkInfo):
                self.lastestNetworkInfo = message

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
                if not self.checkVersion():
                    time.sleep(0.1)
                    continue

                actionMask = np.ones(self.env.actionSpace)
                while True:
                    action = self.algo.getAction(
                        self.network, state, actionMask, isTraining)
                    nextState, reward, done = self.env.takeAction(action.index)
                    if not (nextState == state).all():
                        break
                    actionMask[action.index] = 0
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

    def broadcast(self, message):
        for evaluator in self.evaluators:
            evaluator.send(message)

    def run(self, train: bool = True, episodes: int = 10000, delay: float = 0) -> None:
        self.delay = delay
        self.isTraining = train
        self.lastPrint = time.perf_counter()
        self.totalEpisodes = 0
        self.totalSteps = 0
        self.epochs = 1
        self.dropped = 0

        # mp.set_start_method("spawn")
        # Create Evaluators
        print(
            f"Train: {self.isTraining}, Total Episodes: {self.totalEpisodes}, Total Steps: {self.totalSteps}")
        evaluators = []
        # n_workers = mp.cpu_count() - 1
        # n_workers = mp.cpu_count() // 2
        n_workers = 4
        for i in range(n_workers):
            child = Child(i, self.createEvaluator).start()
            evaluators.append(child)

        trainer = Trainer(self.algo, self.env)

        lastBroadcast = None
        self.evaluators = np.array(evaluators)
        self.epoch = Epoch(episodes).start()
        while True:
            # print("Evaluators Poll")
            for evaluator in self.evaluators:
                while evaluator.poll():
                    message = evaluator.recv()
                    if isinstance(message, MemoryPush):
                        # print("Trainer: Received Memory Push")
                        steps = len(message.memory)
                        if message.version >= trainer.network.version - self.algo.policy.versionTolerance:
                            loss = trainer.learn(message.memory)
                            self.epoch.trained(loss, steps)
                            self.totalSteps += steps
                            if self.epoch.isEnd:
                                self.update()
                                print()
                                self.epoch = Epoch(episodes).start()
                                self.epochs += 1
                        else:
                            self.epoch.drops += steps
                    elif isinstance(message, EnvReport):
                        self.epoch.add(message)
                    else:
                        raise Exception("Unknown Message")

            networkInfo = trainer.network.getInfo()
            if lastBroadcast is None or lastBroadcast.version < networkInfo.version:
                self.broadcast(networkInfo)
                lastBroadcast = networkInfo

            if time.perf_counter() - self.lastPrint > .1:
                self.update()

    def update(self) -> None:
        print(f"#{self.epochs} {Function.humanize(self.epoch.episodes):>6} {self.epoch.hitRate:>7.2%} | " +
              f'Loss: {Function.humanize(self.epoch.loss):>6}/ep | ' +
              f'Best: {Function.humanize(self.epoch.bestRewards):>6}, Avg: {Function.humanize(self.epoch.avgRewards):>6} | ' +
              f'Steps: {Function.humanize(self.epoch.steps / self.epoch.duration):>5}/s | Episodes: {1 / self.epoch.durationPerEpisode:>6.2f}/s | ' +
              f' {Function.humanizeTime(self.epoch.duration):>5} > {Function.humanizeTime(self.epoch.estimateDuration):}' +
              '                                 ',
              end="\b\r")
        self.lastPrint = time.perf_counter()

    def createEvaluator(self, conn):
        Evaluator(self.algo, self.env, conn).start(self.isTraining)

    def createTrainer(self, conn):
        Trainer(self.algo, self.env, conn).start()


class Child:
    def __init__(self, id, target, args=()):
        self.id = id
        self.process = None
        self.target = target
        self.args = args
        self.conn = None

    def start(self):
        self.conn, child_conn = mp.Pipe(True)
        self.process = mp.Process(
            target=self.target, args=self.args + (child_conn,))
        self.process.start()
        return self

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, object):
        return self.conn.send(object)
