import sys
import time
import numpy as np
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
        
        hidden_nodes = 64
        if type(inputShape) is tuple and len(inputShape) == 3:
            self.body = nn.Sequential(
                nn.Conv2d(inputShape[0], 32, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(1),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(1),
                # nn.Conv2d(64, 64, kernel_size=1, stride=1),
                # nn.ReLU(),
                # nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * inputShape[1] * inputShape[2], hidden_nodes),
                nn.ReLU())
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

    def createMemory(self):
        raise NotImplementedError

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
            versionTolerance=4))
        self.gamma = 0.9
        self.epsClip = 0.2

    def createMemory(self, len):
        return SimpleMemory(len)
        
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
        return loss

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

    def learn(self, memory):
        return self.algo.learn(self.network, memory)

class Evaluator(Base):
    def __init__(self, algo: Algo, env, delay=0):
        super().__init__(algo, env)
        self.delay = delay
        self.network = self.algo.createNetwork(self.env.observationShape, self.env.actionSpace)

    def setCommitListener(self, onCommit):
        self.onCommit = onCommit

    def eval(self, isTraining=False):
        memory = []
        report = EpisodeReport()
        state = self.env.reset()
        done: bool = False
        totalRewards = 0
        while not done:
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
        return report

class Agent:
    def __init__(self, algo: Algo, env):
        self.evaluators = []
        self.algo = algo
        self.env = env
        self.history = []

    def run(self, train: bool = True, episodes: int = 1000, epochs: int = 10000, delay: float = 0) -> None:
        self.delay = delay
        self.isTraining = train
        self.target_episodes = episodes
        self.target_epochs = epochs
        self.lastPrint = time.perf_counter()
        self.totalEpisodes = 0
        self.totalSteps = 0
        self.startTime = time.perf_counter()
        self.episodes = 0
        self.epochs = 1

        # mp.set_start_method("spawn")
        # Create Evaluators
        print(f"Train: {self.isTraining}, Total Episodes: {self.totalEpisodes}, Total Steps: {self.totalSteps}")
        evaluators = []
        n_workers = 6  # mp.cpu_count() // 2
        for i in range(n_workers):
            parent_conn, child_conn = mp.Pipe(True)
            p = mp.Process(target=self.createEvaluator, args=(i, child_conn))
            p.start()
            evaluators.append({
                "process": p,
                "conn": parent_conn
            })
        
        self.evaluators.extend(evaluators)
        trainer = Trainer(self.algo, self.env)
        stateDictCache = None
        stateDiceCacheVersion = -1
        while True:
            for evaluator in self.evaluators:
                while evaluator["conn"].poll():
                    message = evaluator["conn"].recv()
                    if isinstance(message, MemoryPush):
                        # print("Trainer received memory.", len(message.memory))
                        if message.version >= trainer.network.version - trainer.algo.policy.versionTolerance \
                            and len(message.memory) >= 5:
                            loss = trainer.learn(message.memory)
                        else:
                            pass
                            # print("memory is dropped.", message.version, trainer.network.version)
                        # print("learnt", loss)
                    elif isinstance(message, NetworkPull):
                        # print("Trainer received Network Pull")
                        if stateDiceCacheVersion == trainer.network.version:
                            stateDict = stateDictCache
                        else:
                            stateDict = trainer.network.state_dict()
                            for key, value in stateDict.items():
                                stateDict[key] = value.cpu()
                            # print(stateDicts)
                            stateDictCache = stateDict
                            stateDiceCacheVersion = trainer.network.version
                        evaluator["conn"].send(NetworkPush(stateDict, trainer.network.version))
                    elif isinstance(message, EpisodeReport):
                        self.history.append(message)
                        self.episodes += 1
                    else:
                        raise Exception("Unknown Message")

            if time.perf_counter() - self.lastPrint > .1:
                self.update()

    def update(self) -> None:
        duration = time.perf_counter() - self.startTime
        avgLoss = np.mean([x.loss for x in self.history]) if len(self.history) > 0 else math.nan
        bestReward = np.max([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        avgReward = np.mean([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        # stdReward = np.std([x.rewards for x in self.history]) if len(self.history) > 0 else math.nan
        progress = self.episodes / self.target_episodes
        # invalidMovesPerEpisode = np.mean([x.invalidMoves for x in self.history])
        durationPerEpisode = duration / self.episodes if self.episodes > 0 else math.nan
        estimateDuration = self.target_episodes * durationPerEpisode
        totalSteps = np.sum([x.steps for x in self.history])
        # print(f"#{self.epochs} {progress:>4.0%} {humanize.intword(self.totalSteps)} | " +
        #       f'Loss: {avgLoss:6.2f}/ep | Best: {bestReward:>5}, Avg: {avgReward:>5.2f} | ' +
        #       f'Steps: {totalSteps/duration:>7.2f}/s | Episodes: {1/durationPerEpisode:>6.2f}/s | ' +
        #       f'Time: {duration: >4.2f}s > {estimateDuration: >5.2f}s', end="\b\r")
        print(f"#{self.epochs} {self.episodes} {humanize.intword(self.totalSteps)} | " +
              f'Loss: {avgLoss:6.2f}/ep | Best: {bestReward:>5}, Avg: {avgReward:>5.2f} | ' +
              f'Steps: {totalSteps/duration:>7.2f}/s | Episodes: {1/durationPerEpisode:>6.2f}/s | ' +
              f'Time: {duration: >4.2f}s > {estimateDuration: >5.2f}s', end="\b\r")
        self.lastPrint = time.perf_counter()

    def createEvaluator(self, i, conn):
        print(i, "started")
        evaluator = Evaluator(self.algo, self.env)

        def onCommit(memory):
            conn.send(MemoryPush(memory, evaluator.network.version))
            conn.send(NetworkPull())
            message = conn.recv()
            evaluator.network.load_state_dict(message.stateDict)
            evaluator.network.version = message.version
        evaluator.setCommitListener(onCommit)
        while True:
            report = evaluator.eval(self.isTraining)
            conn.send(report)




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
        