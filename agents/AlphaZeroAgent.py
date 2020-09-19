from __future__ import annotations
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
# from .Agent import Agent
from typing import List, Callable, TypeVar, Generic, Tuple, Any
import collections
import numpy as np
from enum import Enum
import time
import sys
import utils.Function as Function
import traceback

from memories.Transition import Transition
from games.GameFactory import GameFactory
from multiprocessing.managers import NamespaceProxy, SyncManager
from multiprocessing.connection import Pipe
from .Agent import Base, EvaluatorService, SyncContext, Action, Config, TrainerProcess, Algo, Evaluator, Role, AlgoHandler
from utils.PipedProcess import Process, PipedProcess
from utils.Normalizer import RangeNormalizer, StdNormalizer
from utils.Message import NetworkInfo, LearnReport, EnvReport
from utils.Network import Network, BodyLayers
from utils.PredictionHandler import PredictionHandler
from utils.KyleList import KyleList
from .mcts import MCTS


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
# torch.set_printoptions(edgeitems=10)
torch.set_printoptions(edgeitems=sys.maxsize)


class AlphaZeroNetwork(Network):
    def __init__(self, inputShape, n_outputs):
        super().__init__(inputShape, n_outputs)

        hidden_nodes = 256
        # semi_hidden_nodes = hidden_nodes // 2
        self.body = BodyLayers(inputShape, hidden_nodes)

        # Define policy head
        self.policy = nn.Linear(self.body.num_output, n_outputs)

        # Define value head
        self.value = nn.Sequential(
            nn.Linear(self.body.num_output, 1),
            nn.Tanh())

        # self.value = nn.Linear(self.body.num_output, 1)
        
        self.initWeights()

    def buildOptimizer(self, learningRate):
        # self.optimizer = optim.SGD(self.parameters(), lr=learningRate * 100, momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        return self

    def _body(self, x):
        x = self.body(x)
        return x

    def _policy(self, x, m=None):
        x = self.policy(x)
        if m is not None:
            x = x.masked_fill(~m, -math.inf)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x, m=None):
        x = self._body(x)
        return self._policy(x, m), self.value(x)

    def getPolicy(self, x, m=None):
        x = self._body(x)
        return self._policy(x, m)

    def getValue(self, x):
        x = self._body(x)
        return self.value(x)

class AlphaZeroConfig(Config):
    def __init__(self, sampleSize=32, batchSize=1024, learningRate=1e-3, simulations=100):
        super().__init__(sampleSize, batchSize, learningRate)
        self.simulations = simulations


class AlphaZeroHandler(AlgoHandler):
    def __init__(self, config, env, role, device):
        super().__init__(config, env, role, device)
        self.network = AlphaZeroNetwork(env.observationShape, env.actionSpace)
        self.network.to(self.device)
        self.mcts = MCTS(self.getProb, n_playout=self.config.simulations)
        if role == Role.Trainer:
            self.optimizer = self.network.buildOptimizer(self.config.learningRate)

    def dump(self):
        data = {}
        data["network"] = self._getStateDict(self.network)
        return data

    def load(self, data):
        self.network.load_state_dict(data["network"])

    def reset(self):
        self.mcts.update_with_move(-1)

    def reportStep(self, action):
        self.mcts.update_with_move(action)
        
    def getProb(self, env) -> Tuple[Action, Any]:
        self.network.eval()
        with torch.no_grad():
            state = env.getState()
            mask = env.getMask(state)
            stateTensor = torch.tensor([state], dtype=torch.float, device=self.device)
            maskTensor = torch.tensor([mask], dtype=torch.bool, device=self.device)
            prob, value = self.network(stateTensor, maskTensor)
            return prob.squeeze(0).cpu().detach().numpy(), \
                value.item()

    def getAction(self, env, isTraining: bool) -> Tuple[Action, Any]:
        acts, probs = self.mcts.get_move_probs(env, isTraining, temp=1)
        if isTraining:
            noise_probs = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            index = np.random.choice(len(probs), p=noise_probs)
        else:
            index = np.argmax(probs)
        prediction = np.zeros(self.env.actionSpace)
        prediction[list(acts)] = probs
        state = env.getState()
        mask = env.getMask(state)
        print(state[0] + state[1] * 2, acts, probs, index, "\n")
        self.mcts.update_with_move(acts[index])
        return Action(
            index=acts[index],
            mask=mask,
            probs=prediction)

    def preprocess(self, memory):
        with torch.no_grad():
            lastValue = 0
            for i in reversed(range(len(memory))):
                transition = memory[i]
                transition.reward = transition.reward + lastValue
                lastValue = transition.reward

    def learn(self, memory):
        self.network.train()
        
        self.network.optimizer.zero_grad()
        batchSize = min(memory.size(), self.config.batchSize)
        n_miniBatch = memory.size() // batchSize
        totalLoss = 0

        for i in range(n_miniBatch):
            startIndex = i * batchSize
            minibatch = memory.get(startIndex, batchSize)

            # Get Tensors
            states = minibatch.select(lambda x: x.state).toTensor(torch.float, self.device)
            masks = minibatch.select(lambda x: x.action.mask).toTensor(torch.bool, self.device)
            batch_probs = minibatch.select(lambda x: x.action.probs).toTensor(torch.float, self.device)
            batch_returns = minibatch.select(lambda x: x.reward).toTensor(torch.float, self.device)

            probs, values = self.network(states, masks)

            values = values.squeeze(-1)
            eps = torch.finfo(torch.float).eps
            log_probs = probs.clamp(eps, 1 - eps).log()

            # minimize policy loss using entropy like formula: mean(sum(-p*logp))
            policy_loss = (-batch_probs * log_probs).sum(-1).mean()
            # print((batch_probs * probs.log()).sum(-1))
            print(batch_returns, values)
            # MSE Loss
            value_loss = (batch_returns - values).pow(2).mean()

            # Calculating Total loss
            # the weight of this minibatch
            weight = len(minibatch) / len(memory)
            loss = (policy_loss + value_loss) * weight
            # print("Loss:", loss, policy_loss, entropy_loss, value_loss, weight)

            # Accumulating the loss to the graph
            loss.backward()
            totalLoss += loss.item()

        # Chip grad with norm
        # https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/ppo2/model.py#L107
        nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)

        self.network.optimizer.step()

        return totalLoss

        
class AlphaZeroAlgo(Algo[AlphaZeroConfig]):
    def __init__(self, config=AlphaZeroConfig()):
        super().__init__("AlphaZero", config)

    def createHandler(self, env, role, device):
        return AlphaZeroHandler(self.config, env, role, device)
