import sys
import time
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
import humanize
# def init_layer(m):
#     weight = m.weight.data
#     weight.normal_(0, 1)
#     weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
#     nn.init.constant_(m.bias.data, 0)
#     return m

class Network(nn.Module):
    def __init__(self, inputShape, n_outputs):
        super().__init__()
        self.optimizer = None
        self.version = 0

    def buildOptimizer(self):
        raise NotImplementedError

class PPONetwork(Network):
    def __init__(self, inputShape, n_outputs):
        super().__init__(inputShape, n_outputs)
        
        hidden_nodes = 128
        if type(inputShape) is tuple and len(inputShape) == 3:
            self.body = nn.Sequential(
                nn.Conv2d(inputShape[0], 32, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(1),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(1),
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(1),
                nn.Flatten(),
                nn.Linear(64 * inputShape[1] * inputShape[2], hidden_nodes),
                nn.ReLU(inplace=True))
        else:
            
            if type(inputShape) is tuple and len(inputShape) == 1:
                inputShape = inputShape[0]

            self.body = nn.Sequential(
                nn.Linear(inputShape, hidden_nodes),
                nn.Tanh(),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.Tanh())
                
        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, n_outputs),
            nn.Softmax(dim=-1))
            
        # Define value head
        self.value = nn.Sequential(
            nn.Linear(hidden_nodes, 1))

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
    def __init__(self, batchSize, learningRate, versionTolerance):
        self.batchSize = batchSize
        self.versionTolerance = versionTolerance
        self.learningRate = learningRate

class OnPolicy(Policy):
    def __init__(self, batchSize, learningRate):
        super().__init__(batchSize, learningRate, 0)
        
class OffPolicy(Policy):
    def __init__(self, batchSize, learningRate):
        super().__init__(batchSize, learningRate, math.inf)

class Algo:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # def createMemory(self):
    #     raise NotImplementedError

    def createNetwork(self) -> Network:
        raise NotImplementedError

    def getPrediction(self, network, state, isTrainig: bool):
        raise NotImplementedError

    def getAction(self, prediction, isTrainig: bool):
        raise NotImplementedError

    def learn(self, network: Network, memory):
        raise NotImplementedError

class PPOAlgo(Algo):
    def __init__(self):
        super().__init__(Policy(
            batchSize=32,
            learningRate=0.001,
            versionTolerance=10))
        self.gamma = 0.9
        self.epsClip = 0.2

    # def createMemory(self, len):
    #     return SimpleMemory(len)
        
    def createNetwork(self, inputShape, n_outputs) -> Network:
        return PPONetwork(inputShape, n_outputs).to(self.device)

    def getPrediction(self, network, state, isTraining: bool):
        network.eval()
        with torch.no_grad():
            state = torch.FloatTensor([state]).to(self.device)
            prediction = network.getPolicy(state).squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, isTraining: bool):
        if isTraining:
            return np.random.choice(len(prediction), p=prediction)
        else:
            return prediction.argmax()

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
            value = values[i].item()
            detlas = rewards[i] + self.gamma * lastValue * (1 - dones[i]) - value
            gae = detlas + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages[i] = gae
            lastValue = value
        return advantages

    def learn(self, network: Network, memory):
        network.train()

        states = np.array([x.state for x in memory])
        states = torch.FloatTensor(states).to(self.device)
        
        actions = np.array([x.action for x in memory])
        actions = torch.LongTensor(actions).to(self.device)
        
        predictions = np.array([x.prediction for x in memory])
        predictions = torch.FloatTensor(predictions).to(self.device)

        dones = np.array([x.done for x in memory])
        # dones = torch.BoolTensor(dones).to(self.device)

        rewards = np.array([x.reward for x in memory])
        nextStates = np.array([x.nextState for x in memory])
        real_probs = torch.distributions.Categorical(probs=predictions).log_prob(actions)

        lastState = torch.FloatTensor([nextStates[-1]]).to(self.device)

        lastValue = 0 if dones[-1] else network.getValue(lastState).item()
        targetValues = self.getDiscountedRewards(rewards, dones, lastValue)
        targetValues = torch.FloatTensor(targetValues).to(self.device)
        
        action_probs, values = network(states)
        values = values.squeeze(1)

        advantages = self.getAdvantages(rewards, dones, values, lastValue)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = Function.normalize(advantages)

        dist = torch.distributions.Categorical(probs=action_probs)
        ratios = torch.exp(dist.log_prob(actions) - real_probs)  # porb1 / porb2 = exp(log(prob1) - log(prob2))
        surr1 = ratios * advantages
        surr2 = ratios.clamp(1 - self.epsClip, 1 + self.epsClip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()  # Maximize Policy Loss
        entropy_loss = -dist.entropy().mean()  # Maximize Entropy Loss
        value_loss = F.mse_loss(values, targetValues)  # Minimize Value Loss
        loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss
        
        network.optimizer.zero_grad()
        loss.backward()
        # Chip grad with norm
        nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
        network.optimizer.step()
        network.version += 1

        # Report
        return loss.item()

class Base:
    def __init__(self, algo: Algo, env):
        self.algo: Algo = algo
        self.env = env

    def init(self):
        raise NotImplementedError

class Trainer(Base):
    def __init__(self, algo: Algo, env):
        super().__init__(algo, env)
        self.network = self.algo.createNetwork(self.env.observationShape, self.env.actionSpace).buildOptimizer(self.algo.policy.learningRate)
        self.stateDictCache = None
        self.stateDictCacheVersion = -1

    def learn(self, memory):
        return self.algo.learn(self.network, memory)

    def getStateDict(self):
        if self.stateDictCacheVersion == self.network.version:
            stateDict = self.stateDictCache
        else:
            stateDict = self.network.state_dict()
            for key, value in stateDict.items():
                stateDict[key] = value.cpu()
            self.stateDictCache = stateDict
            self.stateDictCacheVersion = self.network.version
        return stateDict

class Evaluator(Base):
    def __init__(self, algo: Algo, env, conn, delay=0):
        super().__init__(algo, env.getNew())
        self.delay = delay
        # self.algo.device = torch.device("cpu")
        self.network = self.algo.createNetwork(self.env.observationShape, self.env.actionSpace)
        self.network.version = -1
        self.conn = conn
        self.isRequesting = False

    def pullNetwork(self):
        self.conn.send(NetworkPull())
        message: NetworkPush = self.conn.recv()
        if isinstance(message, NetworkPush):
            if message.version > self.network.version:
                self.network.load_state_dict(message.stateDict)
                self.network.version = message.version

    def requestPullNetwork(self):
        if not self.isRequesting:
            self.conn.send(NetworkPull())
            self.isRequesting = True

    def tryPullNetwork(self):
        if self.isRequesting and self.conn.poll():
            message: NetworkPush = self.conn.recv()
            if isinstance(message, NetworkPush):
                self.isRequesting = False
                if message.version > self.network.version:
                    self.network.load_state_dict(message.stateDict)
                    self.network.version = message.version
                    return True
        
        return False

    def pushMemory(self, memory):
        self.conn.send(MemoryPush(memory, self.network.version))

    def onCommit(self, memory):
        self.pushMemory(memory)
        self.requestPullNetwork()
        # self.pullNetwork()
        
    def eval(self, isTraining=False):
        self.pullNetwork()
        while True:
            memory = collections.deque(maxlen=self.algo.policy.batchSize)
            report = EpisodeReport()
            state = self.env.reset()
            done: bool = False
            while not done:
                if self.tryPullNetwork():
                    memory.clear()
                prediction = self.algo.getPrediction(self.network, state, isTraining)
                action = self.algo.getAction(prediction, isTraining)
                nextState, reward, done = self.env.takeAction(action)
                transition = Transition(state, action, reward, nextState, done, prediction)
                report.rewards += reward
                memory.append(transition)
                if transition.done or len(memory) >= self.algo.policy.batchSize:
                    self.onCommit(memory)
                    memory.clear()
                if self.delay > 0:
                    time.sleep(self.delay)
                state = nextState
            self.conn.send(report)

class Agent:
    def __init__(self, algo: Algo, env):
        self.evaluators = []
        self.algo = algo
        self.env = env
        self.history = []

    def run(self, train: bool = True, episodes: int = 1000, epochs: int = 10000, delay: float = 0) -> None:
        self.delay = delay
        self.isTraining = train
        self.lastPrint = time.perf_counter()
        self.totalEpisodes = 0
        self.totalSteps = 0
        self.epochs = 1
        self.dropped = 0

        mp.set_start_method("spawn")
        # Create Evaluators
        print(f"Train: {self.isTraining}, Total Episodes: {self.totalEpisodes}, Total Steps: {self.totalSteps}")
        evaluators = []
        # n_workers = mp.cpu_count() - 1
        n_workers = mp.cpu_count() // 2
        # n_workers = 3
        for i in range(n_workers):
            child = Child(i, self.createEvaluator).start()
            evaluators.append(child)
        
        self.evaluators.extend(evaluators)
        
        trainer = Trainer(self.algo, self.env)
        self.epoch = Epoch(episodes).start()
        while True:
            for evaluator in self.evaluators:
                while evaluator.poll():
                    message = evaluator.recv()
                    if isinstance(message, MemoryPush):
                        if message.version >= trainer.network.version - trainer.algo.policy.versionTolerance \
                            and len(message.memory) >= 5:
                            loss = trainer.learn(message.memory)
                            self.epoch.trained(loss, len(message.memory))
                            self.totalSteps += len(message.memory)
                        else:
                            self.dropped += len(message.memory)
                            # print("memory is dropped.", message.version, trainer.network.version)
                        # print("learnt", loss)
                    elif isinstance(message, NetworkPull):
                        evaluator.send(NetworkPush(trainer.getStateDict(), trainer.network.version))
                    elif isinstance(message, EpisodeReport):
                        self.epoch.add(message)
                        self.epoch.episodes += 1
                        if self.epoch.episodes >= self.epoch.target_episodes:
                            self.epoch = Epoch(episodes).start()
                            self.epochs += 1
                            print()
                    else:
                        raise Exception("Unknown Message")

            if time.perf_counter() - self.lastPrint > .1:
                self.update()

    def update(self) -> None:
        print(f"#{self.epochs} {self.epoch.episodes} {humanize.intword(self.totalSteps)} | " +
              f'Loss: {self.epoch.loss:6.2f}/ep | ' + 
              f'Best: {self.epoch.bestRewards:>5}, Avg: {self.epoch.avgRewards:>5.2f} | ' +
              f'Steps: {self.epoch.steps / self.epoch.duration:>7.2f}/s | Episodes: {1 / self.epoch.durationPerEpisode:>6.2f}/s | ' +
              f'Dropped: {self.dropped} | ' +
              f'Time: {self.epoch.duration: >4.2f}s > {self.epoch.estimateDuration: >5.2f}s'
              , 
              end="\b\r")
        self.lastPrint = time.perf_counter()

    def createEvaluator(self, conn):
        Evaluator(self.algo, self.env, conn).eval(self.isTraining)


class Child:
    def __init__(self, id, target, args=()):
        self.id = id
        self.process = None
        self.target = target
        self.args = args
        self.conn = None

    def start(self):
        self.conn, child_conn = mp.Pipe(True)
        self.process = mp.Process(target=self.target, args=self.args + (child_conn,))
        self.process.start()
        return self

    def poll(self):
        return self.conn.poll()

    def recv(self):
        return self.conn.recv()

    def send(self, object):
        return self.conn.send(object)


class Message:
    def __init__(self):
        pass

class MemoryPush(Message):
    def __init__(self, memory, version):
        super().__init__()
        self.memory = memory
        self.version = version

class NetworkPull(Message):
    def __init__(self):
        pass

class NetworkPush(Message):
    def __init__(self, stateDict, version):
        self.stateDict = stateDict
        self.version = version

class EpisodeReport(Message):
    def __init__(self):
        self.steps: int = 0
        self.rewards: float = 0
        self.total_loss: float = 0
        self.episode_start_time: int = 0
        self.episode_end_time: int = 0

    def start(self):
        self.episode_start_time = time.perf_counter()
        return self

    def end(self):
        self.episode_end_time = time.perf_counter()
        return self

    @property
    def duration(self):
        return (self.episode_end_time if self.episode_end_time > 0 else time.perf_counter()) - self.episode_start_time

    @property
    def loss(self):
        return self.total_loss / self.steps if self.steps > 0 else 0

    def trained(self, loss, steps):
        self.total_loss += loss * steps
        self.steps += steps
        

class Epoch(Message):
    def __init__(self, target_episodes):
        self.target_episodes = target_episodes
        self.steps: int = 0
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
        return self.duration / self.episodes if self.episodes > 0 else math.nan

    @property
    def estimateDuration(self):
        return self.target_episodes * self.durationPerEpisode

    @property
    def avgRewards(self):
        return self.totalRewards / self.episodes if self.episodes > 0 else math.nan

    def add(self, episode):
        if episode.rewards > self.bestRewards:
            self.bestRewards = episode.rewards
        self.totalRewards += episode.rewards
        self.history.append(episode)
        return self

    def trained(self, loss, steps):
        self.total_loss += loss * steps
        self.steps += steps
        return self
        