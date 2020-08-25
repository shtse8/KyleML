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
import multiprocessing
# def init_layer(m):
#     weight = m.weight.data
#     weight.normal_(0, 1)
#     weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
#     nn.init.constant_(m.bias.data, 0)
#     return m

class Message:
    def __init__(self):
        pass

class LastestInfo(Message):
    def __init__(self, networkStateDict, networkVersion):
        self.networkStateDict = networkStateDict
        self.networkVersion = networkVersion

class MemoryCollected(Message):
    def __init__(self, count=1):
        self.count = count

class MemoryPull(Message):
    def __init__(self):
        super().__init__()

class MemoryPush(Message):
    def __init__(self, memory, version):
        super().__init__()
        self.memory = memory
        self.version = version

class NetworkPull(Message):
    def __init__(self):
        super().__init__()

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
    def isEnd(self):
        return self.epoch_end_time > 0

    @property
    def duration(self):
        return (self.episode_end_time if self.episode_end_time > 0 else time.perf_counter()) - self.episode_start_time

    @property
    def loss(self):
        return self.total_loss / self.steps if self.steps > 0 else 0

    def trained(self, loss, steps):
        self.total_loss += loss * steps
        self.steps += steps
        

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

    def add(self, episode):
        if episode.rewards > self.bestRewards:
            self.bestRewards = episode.rewards
        self.totalRewards += episode.rewards
        self.history.append(episode)
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
        self.version = 0

    def buildOptimizer(self):
        raise NotImplementedError

class PPONetwork(Network):
    def __init__(self, inputShape, n_outputs):
        super().__init__(inputShape, n_outputs)
        
        hidden_nodes = 64
        if type(inputShape) is tuple and len(inputShape) == 3:
            self.body = nn.Sequential(
                nn.Conv2d(inputShape[0], 32, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(1),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(1),
                nn.Conv2d(64, 64, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(1),
                nn.Flatten(),
                nn.Linear(64 * inputShape[1] * inputShape[2], hidden_nodes),
                nn.ReLU(inplace=True))
        else:
            
            if type(inputShape) is tuple and len(inputShape) == 1:
                inputShape = inputShape[0]

            self.body = nn.Sequential(
                nn.Linear(inputShape, hidden_nodes),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_nodes, hidden_nodes),
                nn.ReLU(inplace=True))
                
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

    def getAction(self, network, state, mask, isTraining: bool) -> PredictedAction:
        raise NotImplementedError

    def learn(self, network: Network, memory):
        raise NotImplementedError

class PPOAlgo(Algo):
    def __init__(self):
        super().__init__(Policy(
            batchSize=32,
            learningRate=0.0001,
            versionTolerance=0))
        self.gamma = 0.9
        self.epsClip = 0.2

    # def createMemory(self, len):
    #     return SimpleMemory(len)
        
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
            detlas = rewards[i] + self.gamma * lastValue * (1 - dones[i]) - values[i]
            gae = detlas + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages[i] = gae
            lastValue = values[i]
        return advantages

    def processMemory(self, network, memory):
        dones = np.array([x.done for x in memory])
        rewards = np.array([x.reward for x in memory])
        lastValue = 0
        lastMemory = memory[-1]
        if not lastMemory.done:
            lastState = torch.FloatTensor([lastMemory.nextState]).to(self.device)
            lastValue = network.getValue(lastState).item()
        newRewards = self.getDiscountedRewards(rewards, dones, lastValue)
        for i, transition in enumerate(memory):
            transition.reward = newRewards[i]

    def learn(self, network: Network, memory):
        network.train()

        states = np.array([x.state for x in memory])
        states = torch.FloatTensor(states).to(self.device)

        actions = np.array([x.action.index for x in memory])
        actions = torch.LongTensor(actions).to(self.device)
        
        predictions = np.array([x.action.prediction for x in memory])
        predictions = torch.FloatTensor(predictions).to(self.device)
        real_probs = torch.distributions.Categorical(probs=predictions).log_prob(actions)

        # dones = np.array([x.done for x in memory])
        rewards = np.array([x.reward for x in memory])
        rewards = torch.FloatTensor(rewards).to(self.device)
        

        # lastValue = 0
        # if not dones[-1]:
        #     nextStates = np.array([x.nextState for x in memory])
        #     lastState = torch.FloatTensor([nextStates[-1]]).to(self.device)
        #     lastValue = network.getValue(lastState).item()
        # targetValues = self.getDiscountedRewards(rewards, dones, lastValue)
        # targetValues = torch.FloatTensor(targetValues).to(self.device)
        # targetValues = Function.normalize(targetValues)

        lossList = []
        for _ in range(1):

            action_probs, values = network(states)
            values = values.squeeze(1)

            advantages = rewards - values
            # print(targetValues, values)
            # print(advantages)
            # advantages = self.getAdvantages(rewards, dones, values.cpu().detach().numpy(), lastValue)
            # advantages = torch.FloatTensor(advantages).to(self.device)
            # print(advantages)
            # targetValues = advantages + values
            # advantages = Function.normalize(advantages)

            dist = torch.distributions.Categorical(probs=action_probs)
            ratios = torch.exp(dist.log_prob(actions) - real_probs)  # porb1 / porb2 = exp(log(prob1) - log(prob2))
            surr1 = ratios * advantages
            surr2 = ratios.clamp(1 - self.epsClip, 1 + self.epsClip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()  # Maximize Policy Loss (Rewards)
            # entropy_loss = -dist.entropy().mean()  # Maximize Entropy Loss
            value_loss = F.mse_loss(values, rewards)  # Minimize Value Loss (Distance to Target)
            loss = policy_loss + 2 * value_loss
            # loss = 0.01 * entropy_loss + 1 * value_loss
            # print(policy_loss, entropy_loss, value_loss, loss)
            
            network.optimizer.zero_grad()
            loss.backward()
            # Chip grad with norm
            # nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 10)
            network.optimizer.step()
            
            lossList.append(loss.item())
        # Report
        network.version += 1
        return np.mean(lossList)

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

    def updateStateDict(self):
        if self.stateDictCacheVersion != self.network.version:
            stateDict = self.network.state_dict()
            for key, value in stateDict.items():
                stateDict[key] = value.cpu().detach().numpy()
            self.stateDictCache = stateDict
            self.stateDictCacheVersion = self.network.version

    def getStateDict(self):
        self.updateStateDict()
        return self.stateDictCache


class Evaluator(Base):
    def __init__(self, algo: Algo, env, conn, delay=0):
        super().__init__(algo, env.getNew())
        self.delay = delay
        # self.algo.device = torch.device("cpu")
        self.network = self.algo.createNetwork(self.env.observationShape, self.env.actionSpace)
        self.network.version = -1
        self.conn = conn
        self.isRequesting = False
        self.memory = collections.deque(maxlen=self.algo.policy.batchSize)
        self.report = None

        self.shouldRun = False

    def waitForNewNetwork(self):
        message = self.conn.recv()
        if isinstance(message, LastestInfo):
            # print("Received new network")
            stateDict = message.networkStateDict
            for key, value in stateDict.items():
                stateDict[key] = torch.from_numpy(value)
            self.network.load_state_dict(stateDict)
            self.network.version = message.networkVersion
            self.memory.clear()
            self.shouldRun = True

    def pushMemory(self):
        if len(self.memory) > 0:
            self.algo.processMemory(self.network, self.memory)
            self.conn.send(MemoryPush(self.memory, self.network.version))
            self.memory.clear()

    def commit(self, transition: Transition):
        self.report.rewards += transition.reward
        self.memory.append(transition)

        self.conn.send(MemoryCollected())
        message = self.conn.recv()
        if isinstance(message, MemoryPull):
            # print("Received memory pull")
            self.pushMemory()
            # wait for new network
            self.waitForNewNetwork()

    def eval(self, isTraining=False):
        self.waitForNewNetwork()
        while True:
            self.report = EpisodeReport().start()
            state = self.env.reset()
            done: bool = False
            while not done:
                # make a new step on env
                actionMask = np.ones(self.env.actionSpace)
                while True:
                    action = self.algo.getAction(self.network, state, actionMask, isTraining)
                    nextState, reward, done = self.env.takeAction(action.index)
                    if not (nextState == state).all():
                        break
                    actionMask[action.index] = 0
                transition = Transition(state, action, reward, nextState, done)

                # commit to memory
                self.commit(transition)
                if self.delay > 0:
                    time.sleep(self.delay)
                state = nextState
            self.conn.send(self.report.end())

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

        mp.set_start_method("spawn")
        # Create Evaluators
        print(f"Train: {self.isTraining}, Total Episodes: {self.totalEpisodes}, Total Steps: {self.totalSteps}")
        evaluators = []
        # n_workers = mp.cpu_count() - 1
        n_workers = mp.cpu_count() // 2
        n_workers = 4
        for i in range(n_workers):
            child = Child(i, self.createEvaluator).start()
            evaluators.append(child)
        
        self.evaluators = np.array(evaluators)


        pulling = False
        lastBoardcast = -1
        trainer = Trainer(self.algo, self.env)
        self.epoch = Epoch(episodes).start()
        memory = collections.deque(maxlen=self.algo.policy.batchSize)
        message = LastestInfo(trainer.getStateDict(), trainer.network.version)
        self.broadcast(message)
        lastLearn = time.perf_counter()
        totalMemoryCount = 0
        while True:
            # conns = multiprocessing.connection.wait([e.conn for e in self.evaluators], 0)
            for evaluator in self.evaluators:
                while evaluator.poll():
                    message = evaluator.recv()
                    if isinstance(message, MemoryCollected):
                        totalMemoryCount += message.count
                        evaluator.n_memory += message.count
                        # print("MemoryCollected", evaluator.id, totalMemoryCount, evaluator.n_memory)
                        # print(evaluator.id, evaluator.n_memory)
                        # totalMemoryCount = np.sum([e.n_memory for e in self.evaluators])
                        if totalMemoryCount >= self.algo.policy.batchSize:
                            if not pulling:
                                pulling = True
                                for e in self.evaluators:
                                    e.pulling = True
                            # print("pull", evaluator.id, evaluator.n_memory)
                            evaluator.send(MemoryPull())
                        else:
                            evaluator.send(Message())

                    elif isinstance(message, MemoryPush):
                        oldLen = len(memory)
                        if len(memory) < self.algo.policy.batchSize and message.version == trainer.network.version:
                            memory.extend(message.memory)
                        # print(evaluator.id, evaluator.n_memory, len(message.memory), oldLen, len(memory), message.version == trainer.network.version)
                        evaluator.n_memory = 0
                        evaluator.pulling = False
                        if not any([e.pulling for e in self.evaluators]):
                            pulling = False
                    elif isinstance(message, EpisodeReport):
                        self.epoch.add(message)
                    else:
                        raise Exception("Unknown Message")

            # Boardcast
            # if not pulling:
            #     totalMemoryCount = np.sum([e.n_memory for e in self.evaluators])
            #     if totalMemoryCount > self.algo.policy.batchSize:
            #         pulling = True
            #         # print("totalMemoryCount", totalMemoryCount)
            #         self.broadcast(MemoryPull())
            # totalMemoryCount = np.array([e.pulling for e in self.evaluators]).all()
            # print([e.pulling for e in self.evaluators])
            if not pulling and len(memory) >= self.algo.policy.batchSize:
                # print("Collect", time.perf_counter() - lastLearn)
                tic = time.perf_counter()
                pulling = False
                # print("Learn")
                # tic2 = time.perf_counter()
                loss = trainer.learn(memory)
                # print(time.perf_counter() - tic2)
                self.epoch.trained(loss, len(memory))
                self.totalSteps += len(memory)
                memory.clear()
                totalMemoryCount = 0
                # print("memory clear")
                if self.epoch.isEnd:
                    self.update()
                    print()
                    self.epoch = Epoch(episodes).start()
                    self.epochs += 1
                message = LastestInfo(trainer.getStateDict(), trainer.network.version)
                self.broadcast(message)
                lastLearn = time.perf_counter()
                # print("Learn", time.perf_counter() - tic)


            if time.perf_counter() - self.lastPrint > .1:
                self.update()

    def update(self) -> None:
        hitRate = 1 - self.dropped / self.totalSteps if self.totalSteps > 0 else math.nan
        print(f"#{self.epochs} {Function.humanize(self.epoch.episodes):>6} {self.epoch.hitRate:>7.2%} | " +
              f'Loss: {Function.humanize(self.epoch.loss):>6}/ep | ' + 
              f'Best: {Function.humanize(self.epoch.bestRewards):>6}, Avg: {Function.humanize(self.epoch.avgRewards):>6} | ' +
              f'Steps: {Function.humanize(self.epoch.steps / self.epoch.duration):>5}/s | Episodes: {1 / self.epoch.durationPerEpisode:>6.2f}/s | ' +
              f'Time: {Function.humanizeTime(self.epoch.duration):>5} > {Function.humanizeTime(self.epoch.estimateDuration):}'
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
        self.n_memory = 0
        self.pulling = False

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

