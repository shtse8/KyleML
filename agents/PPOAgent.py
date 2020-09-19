from __future__ import annotations
import inspect
import asyncio
import __main__
import types
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
from typing import List, Callable, TypeVar, Generic, Tuple, Any
import collections
import numpy as np
from enum import Enum
import time
import sys

from memories.Transition import Transition
from memories.SimpleMemory import SimpleMemory
from games.GameFactory import GameFactory
from multiprocessing.managers import NamespaceProxy, SyncManager
from multiprocessing.connection import Pipe
from .Agent import Base, EvaluatorService, SyncContext, Action, Config
from utils.PipedProcess import Process, PipedProcess
from utils.Normalizer import RangeNormalizer, StdNormalizer
from utils.Message import NetworkInfo, LearnReport, EnvReport
from utils.Network import Network
from utils.multiprocessing import Proxy
from utils.PredictionHandler import PredictionHandler

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
# torch.set_printoptions(edgeitems=10)
torch.set_printoptions(edgeitems=sys.maxsize)


class ICMNetwork(Network):
    def __init__(self, inputShape, output_size):
        super().__init__(inputShape, output_size)

        hidden_nodes = 256
        self.output_size = output_size

        self.feature = nn.Sequential(
            BodyLayers(inputShape, hidden_nodes),
            nn.Linear(hidden_nodes, hidden_nodes))
        
        self.inverseNet = nn.Sequential(
            nn.Linear(hidden_nodes * 2, hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes, output_size)
        )
        
        self.forwardNet = nn.Sequential(
            nn.Linear(output_size + hidden_nodes, hidden_nodes),
            nn.LeakyReLU(),
            nn.Linear(hidden_nodes, hidden_nodes),
        )

        self.initWeights()

    def buildOptimizer(self, learningRate):
        # self.optimizer = optim.SGD(self.parameters(), lr=learningRate * 100, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        return self

    def forward(self, state, next_state, action):
        action = F.one_hot(action, self.output_size).to(action.device)
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        pred_action = self.inverseNet(torch.cat((encode_state, encode_next_state), 1))
        pred_next_state_feature = self.forwardNet(torch.cat((encode_state, action), 1))
        return encode_next_state, pred_next_state_feature, pred_action

class PPONetwork(Network):
    def __init__(self, inputShape, n_outputs):
        super().__init__(inputShape, n_outputs)

        self.eps = torch.finfo(torch.float).eps
        hidden_nodes = 256
        # semi_hidden_nodes = hidden_nodes // 2
        self.body = BodyLayers(inputShape, hidden_nodes)

        self.gru = GRULayers(self.body.num_output, hidden_nodes, 
            num_layers=2, bidirectional=True, dropout=0.2)

        # Define policy head
        self.policy = nn.Linear(self.gru.num_output, n_outputs)

        # Define value head
        self.value = nn.Linear(self.gru.num_output, 1)

        self.initWeights()

    def buildOptimizer(self, learningRate):
        # self.optimizer = optim.SGD(self.parameters(), lr=learningRate * 100, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        return self

    def getInitHiddenState(self, device):
        return self.gru.getInitHiddenState(device)

    def _body(self, x, h):
        x = self.body(x)
        x, h = self.gru(x, h)
        return x, h

    def _policy(self, x, m=None):
        x = self.policy(x)
        if m is not None:
            x = x.masked_fill(~m, -math.inf)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x, h, m=None):
        x, h = self._body(x, h)
        return self._policy(x, m), self.value(x), h

    def getPolicy(self, x, h, m=None):
        x, h = self._body(x, h)
        return self._policy(x, m), h

    def getValue(self, x, h):
        x, h = self._body(x, h)
        return self.value(x), h



class PPOConfig(Config):
    def __init__(self, sampleSize=256, batchSize=1024, learningRate=1e-4, gamma=0.99, epsClip=0.2, gaeCoeff=0.95):
        super().__init__(sampleSize, batchSize, learningRate)
        self.gamma = gamma
        self.epsClip = epsClip
        self.gaeCoeff = gaeCoeff


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


class Iterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.current = 0

    def __iter__(self) -> Iterator:
        return Iterator(self.iterable)

    def __next__(self):
        if self.current < len(self.iterable):
            result = self.iterable[self.current]
            self.current += 1
            return result
        self.current = 0
        raise StopIteration

T = TypeVar('T')
S = TypeVar('S')


class Memory(Generic[T]):
    def __init__(self, iter: List[T]) -> None:
        self.memory = np.array(iter)

    def select(self, property: Callable[[T], S]) -> Memory[S]:
        return Memory([property(x) for x in self.memory])

    def toArray(self) -> np.array:
        return self.memory

    def toTensor(self, dtype: torch.dtype, device: torch.device) -> torch.tensor:
        return torch.tensor(self.memory, dtype=dtype, device=device).detach()

    def get(self, fromPos: int, num: int) -> Memory[T]:
        return Memory(self.memory[fromPos:fromPos+num])

    def mean(self) -> float:
        return self.memory.mean()

    def std(self) -> float:
        return self.memory.std()

    def size(self) -> int:
        return len(self.memory)

    def __sub__(self, other):
        if isinstance(other, Memory):
            other = other.memory
        return Memory(self.memory - other)

    def __truediv__(self, other):
        return Memory(self.memory / other)

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, i: int) -> T:
        return self.memory[i]

    def __iter__(self) -> Iterator:
        return Iterator(self)

    def __str__(self):
        return str(self.memory)



class PPOAlgo(Algo[PPOConfig]):
    def __init__(self, config=PPOConfig()):
        super().__init__("PPO", config)

    def createNetwork(self, inputShape, n_outputs) -> Network:
        return PPONetwork(inputShape, n_outputs)

    def createICMNetwork(self, inputShape, n_outputs) -> Network:
        return ICMNetwork(inputShape, n_outputs)

    def getAction(self, network, state, mask, isTraining: bool, hiddenState=None) -> Tuple[Action, Any]:
        network.eval()
        try:
            with torch.no_grad():
                stateTensor = torch.tensor([state], dtype=torch.float, device=self.device)
                maskTensor = torch.tensor([mask], dtype=torch.bool, device=self.device)
                prediction, nextHiddenState = network.getPolicy(stateTensor, hiddenState.unsqueeze(0), maskTensor)
                dist = torch.distributions.Categorical(probs=prediction)
                index = dist.sample() if isTraining else dist.probs.argmax(dim=-1, keepdim=True)
                return Action(
                    index=index.item(),
                    mask=mask,
                    prediction=prediction.squeeze(0).cpu().detach().numpy()
                ), nextHiddenState.squeeze(0)
        except Exception as e:
            print(e)
            print(prediction)
            sys.exit(0)

    def preprocess(self, network, memory, rewardNormalizer):
        with torch.no_grad():
            lastValue = 0
            lastMemory = memory[-1]
            if not lastMemory.done:
                lastState = Memory([lastMemory.nextState]).toTensor(torch.float, self.device)
                hiddenState = Memory([lastMemory.nextHiddenState]).toTensor(torch.float, self.device)
                lastValue, _ = network.getValue(lastState, hiddenState)
                lastValue = lastValue.item()
            
            states = memory.select(lambda x: x.state).toTensor(torch.float, self.device)
            hiddenStates = memory.select(lambda x: x.hiddenState).toTensor(torch.float, self.device)
            values, _ = network.getValue(states, hiddenStates)
            values = values.squeeze(-1).cpu().detach().numpy()

            returns = memory.select(lambda x: x.reward).toArray()
            # returns = rewardNormalizer.normalize(returns, update=True)
            
            # GAE (General Advantage Estimation)
            # Paper: https://arxiv.org/abs/1506.02438
            # Code: https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L55-L64
            gae = 0
            for i in reversed(range(len(memory))):
                transition = memory[i]
                reward = returns[i]
                detlas = reward + self.config.gamma * lastValue * (1 - transition.done) - values[i]
                gae = detlas + self.config.gamma * self.config.gaeCoeff * gae * (1 - transition.done)
                # from baseline
                # https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L65
                transition.advantage = gae
                transition.reward = gae + values[i]
                transition.value = values[i]
                lastValue = values[i]

    def learn(self, network: Network, icm: Network, memory: Memory[Transition], rewardNormalizer):
        network.train()
        icm.train()
        
        network.optimizer.zero_grad()
        icm.optimizer.zero_grad()
        batchSize = min(memory.size(), self.config.batchSize)
        n_miniBatch = memory.size() // batchSize
        totalLoss = 0

        # Normalize advantages
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L139
        # advantages = memory.select(lambda x: x.reward) - memory.select(lambda x: x.value)
        # We need to normalize the reward to prevent the GRU becomes all 1 and -1.
        
        # returns = memory.select(lambda x: x.reward)
        # rewardNormalizer.update(returns.toArray())
        # print(rewardNormalizer.mean, rewardNormalizer.var, rewardNormalizer.count)
        # returns = rewardNormalizer.normalize(returns)
        # returns = Function.normalize(returns)
        # print(returns)
        advantages = memory.select(lambda x: x.advantage)
        advantages = Function.normalize(advantages)

        for i in range(n_miniBatch):
            startIndex = i * batchSize
            minibatch = memory.get(startIndex, batchSize)

            # Get Tensors
            states = minibatch.select(lambda x: x.state).toTensor(torch.float, self.device)
            # nextStates = minibatch.select(lambda x: x.nextState).toTensor(torch.float, self.device)
            actions = minibatch.select(lambda x: x.action.index).toTensor(torch.long, self.device)
            masks = minibatch.select(lambda x: x.action.mask).toTensor(torch.bool, self.device)
            old_log_probs = minibatch.select(lambda x: x.action.log).toTensor(torch.float, self.device)
            batch_returns = minibatch.select(lambda x: x.reward).toTensor(torch.float, self.device)
            # batch_returns = returns.get(startIndex, batchSize).toTensor(torch.float, self.device)
            # batch_values = minibatch.select(lambda x: x.value).toTensor(torch.float, self.device)
            hiddenStates = minibatch.select(lambda x: x.hiddenState).toTensor(torch.float, self.device)
            # print("hiddenStates:", hiddenStates[0])
            
            icm_loss = 0
            # for Curiosity Network
            # eta = 0.01
            # real_next_state_feature, pred_next_state_feature, pred_action = icm(states, nextStates, actions)
            
            # inverse_loss = nn.CrossEntropyLoss()(pred_action, actions)
            # forward_loss = nn.MSELoss()(pred_next_state_feature, real_next_state_feature.detach())
            # icm_loss = inverse_loss + forward_loss
            # intrinsic_rewards = 0.01 * (real_next_state_feature.detach() - pred_next_state_feature.detach()).pow(2).mean(-1)
            # batch_returns = batch_returns + intrinsic_rewards
            
            # for AC Network
            batch_advantages = advantages.get(startIndex, batchSize).toTensor(torch.float, self.device)
            # batch_advantages = batch_returns - batch_values
            probs, values, _ = network(states, hiddenStates, masks)
            values = values.squeeze(-1)
            # print("returns:", batch_returns)
            # print("values:", values)
            # PPO2 - Confirm the samples aren't too far from pi.
            # porb1 / porb2 = exp(log(prob1) - log(prob2))
            dist = torch.distributions.Categorical(probs=probs)
            ratios = torch.exp(dist.log_prob(actions) - old_log_probs)
            policy_losses1 = ratios * batch_advantages
            policy_losses2 = ratios.clamp(1 - self.config.epsClip, 1 + self.config.epsClip) * batch_advantages

            # Maximize Policy Loss (Rewards)
            policy_loss = -torch.min(policy_losses1, policy_losses2).mean()

            # Maximize Entropy Loss
            entropy_loss = -dist.entropy().mean()

            # Minimize Value Loss  (MSE)

            # Clip the value to reduce variability during Critic training
            # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L66-L75
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py#L69-L75
            # Remark: as the clip is an absolute value, it's not much useful on different large scale scenario
            # value_loss1 = (batch_returns - values).pow(2)
            # valuesClipped = batch_values + \
            #     torch.clamp(values - batch_values, -
            #                 self.config.epsClip, self.config.epsClip)
            # value_loss2 = (batch_returns - valuesClipped).pow(2)
            # value_loss = torch.max(value_loss1, value_loss2).mean()
            
            # MSE Loss
            value_loss = (batch_returns - values).pow(2).mean()
        
            network_loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss

            # Calculating Total loss
            # the weight of this minibatch
            weight = len(minibatch) / len(memory)
            loss = (network_loss + icm_loss) * weight
            # print("Loss:", loss, policy_loss, entropy_loss, value_loss, weight)

            # Accumulating the loss to the graph
            loss.backward()
            totalLoss += loss.item()

        # Chip grad with norm
        # https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/ppo2/model.py#L107
        nn.utils.clip_grad.clip_grad_norm_(network.parameters(), 0.5)
        nn.utils.clip_grad.clip_grad_norm_(icm.parameters(), 0.5)

        network.optimizer.step()
        icm.optimizer.step()
        network.version += 1

        return totalLoss

class Base:
    def __init__(self, algo, gameFactory, sync):
        self.algo = algo
        self.algo.device = sync.getDevice()
        self.gameFactory = gameFactory
        self.sync = sync
        self.networks = []
        self.weightPath = "./weights/"
        self.lastSave = 0

    def save(self) -> None:
        try:
            path = self.getSavePath(True)
            data = {
                "totalSteps": self.sync.totalSteps.value,
                "totalEpisodes": self.sync.totalEpisodes.value
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
            self.sync.totalSteps.value = int(
                data["totalSteps"]) if "totalSteps" in data else 0
            self.sync.totalEpisodes.value = int(
                data["totalEpisodes"]) if "totalEpisodes" in data else 0
            for network in self.networks:
                print(f"{network.name} weights loaded.")
                network.load_state_dict(data[network.name])
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


class Trainer(Base):
    def __init__(self, algo: Algo, gameFactory: GameFactory, sync):
        super().__init__(algo, gameFactory, sync)
        self.evaluators: List[EvaluatorService] = []
        self.network = None
        self.icm = None
        self.rewardNormalizer = StdNormalizer()

    def learn(self, experience):
        steps = len(experience)
        for _ in range(5):
            loss = self.algo.learn(self.network, self.icm, memory, self.rewardNormalizer)
            # learn report handling
            self.sync.epochManager.trained(loss, steps)
            self.sync.totalEpisodes.value += 1
            self.sync.totalSteps.value += steps

    def pushNewNetwork(self):
        networkInfo = self.network.getInfo()
        if networkInfo.version > self.sync.latestVersion.value:
            self.sync.latestStateDict.update(networkInfo.stateDict)
            self.sync.latestVersion.value = networkInfo.version

    async def start(self, episodes=1000, load=False):
        env = self.gameFactory.get()

        # Create Evaluators
        evaluators = []
        n_workers = max(torch.cuda.device_count(), 1)
        for _ in range(n_workers):
            evaluator = EvaluatorService(
                self.algo, self.gameFactory, self.sync).start()
            evaluators.append(evaluator)

        self.evaluators = np.array(evaluators)

        self.network = self.algo.createNetwork(
            env.observationShape, env.actionSpace).buildOptimizer(
            self.algo.config.learningRate).to(self.algo.device)
        self.icm = self.algo.createICMNetwork(
            env.observationShape, env.actionSpace).buildOptimizer(
            self.algo.config.learningRate).to(self.algo.device)
        self.networks.append(self.network)
        self.networks.append(self.icm)
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
                    e = Memory[Transition](e)
                    self.algo.preprocess(self.network, e, self.rewardNormalizer)
                    experience.extend(e)
            # print("learn")
            # learn
            experience = Memory[Transition](experience)
            self.learn(experience)

            if time.perf_counter() - self.lastSave > 60:
                self.save()


class Evaluator(Base):
    def __init__(self, algo: Algo, gameFactory, sync):
        super().__init__(algo, gameFactory, sync)
        self.env = gameFactory.get()
        # self.algo.device = torch.device("cpu")
        # self.algo.device = sync.getDevice()
        self.network = self.algo.createNetwork(
            self.env.observationShape, self.env.actionSpace).to(self.algo.device)
        self.network.version = -1
        self.networks.append(self.network)

        self.playerCount = self.env.getPlayerCount()
        self.agents = np.array(
            [Agent(i + 1, self.env, self.network, algo) for i in range(self.playerCount)])
        self.started = False

        self.reports: List[EnvReport] = []

    def updateNetwork(self):
        if self.network.version < self.sync.latestVersion.value:
            networkInfo = NetworkInfo(
                self.sync.latestStateDict, self.sync.latestVersion.value)
            self.network.loadInfo(networkInfo)

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
                                row, col = map(int, input(
                                    "Your action: ").split())
                                pos = row * self.env.size + col
                                player.step(pos)
                                break
                            except Exception as e:
                                print(e)
                else:
                    agent.step(False)
                # os.system('cls')
                print(self.env.getState(1).astype(int), "\n")
                await asyncio.sleep(0.5)
            if self.env.isDone():
                print("Done:")
                for agent in self.agents:
                    print(agent.id, self.env.getDoneReward(agent.id))
                await asyncio.sleep(3)
            self.flushReports()


class Agent:
    def __init__(self, id, env, network, algo):
        self.id = id
        self.env = env
        self.memory = None
        self.report = EnvReport()
        self.network = network
        self.algo = algo
        self.player = self.env.getPlayer(self.id)
        self.hiddenState = self.network.getInitHiddenState(self.algo.device)

    def step(self, isTraining=True) -> None:
        if not self.env.isDone() and self.player.canStep():
            state = self.player.getState()
            mask = self.player.getMask(state)
            hiddenState = self.hiddenState

            action, nextHiddenState = self.algo.getAction(
                self.network, state, mask, isTraining, hiddenState)
            # if (hiddenState == 0).all():
            #     print("Step:", hiddenState, nextHiddenState)
            # print("Action:", action.index, action.prediction[action.index])
            nextState, reward, done = self.player.step(action.index)
            if self.memory is not None:
                transition = Transition(
                    state=state,
                    hiddenState=hiddenState.cpu().detach().numpy(),
                    action=action,
                    reward=reward,
                    nextState=nextState,
                    nextHiddenState=nextHiddenState.cpu().detach().numpy(),
                    done=done)
                self.memory.append(transition)
            self.hiddenState = nextHiddenState
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
        self.hiddenState = self.network.getInitHiddenState(self.algo.device)
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

