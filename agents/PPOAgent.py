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
import pickle

from memories.Transition import Transition
from memories.SimpleMemory import SimpleMemory
from games.GameFactory import GameFactory
from multiprocessing.managers import NamespaceProxy, SyncManager
from multiprocessing.connection import Pipe
from .Agent import Base, EvaluatorService, SyncContext, Action, Config, Role, AlgoHandler, Algo, TensorWrapper
from utils.PipedProcess import Process, PipedProcess
from utils.Normalizer import RangeNormalizer, StdNormalizer
from utils.Message import NetworkInfo, LearnReport, EnvReport
from utils.Network import Network, BodyLayers, GRULayers
from utils.multiprocessing import Proxy
from utils.PredictionHandler import PredictionHandler
from utils.KyleList import KyleList

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
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
            nn.ELU(),
            nn.Linear(hidden_nodes, output_size)
        )
        
        self.forwardNet = nn.Sequential(
            nn.Linear(output_size + hidden_nodes, hidden_nodes),
            nn.ELU(),
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

class AgentHandler:
    def __init__(self, handler, env):
        self.handler = handler
        self.env = env
        self.hiddenState = self.handler.network.getInitHiddenState(self.handler.device)

    def reset(self):
        self.hiddenState = self.handler.network.getInitHiddenState(self.handler.device)

    def reportStep(self, action):
        self.handler.reportStep(action)

    def getInfo(self):
        info = self.env.getInfo()
        info.hiddenState = self.hiddenState.cpu().detach().numpy()
        return info

    def getProb(self):
        self.handler.network.eval()
        with torch.no_grad():
            info = self.env.getInfo()
            state = KyleList([info.state]).toTensor(torch.float, self.handler.device)
            mask = KyleList([info.mask]).toTensor(torch.bool, self.handler.device)
            prob, value, hiddenState = self.handler.network(state, self.hiddenState.unsqueeze(0), mask)
            return TensorWrapper(prob.squeeze(0)), TensorWrapper(value), TensorWrapper(hiddenState.squeeze(0))

    def getAction(self, isTraining: bool) -> Tuple[Action, Any]:
        probs, value, hiddenState = self.getProb()
        self.hiddenState = hiddenState.asTensor()
        if isTraining:
            index = np.random.choice(len(probs), p=probs.asArray())
        else:
            index = np.argmax(probs.asArray())
        return Action(
            index=index,
            probs=probs.asArray(),
            value=value.asItem())



class PPOHandler(AlgoHandler):
    def __init__(self, config, env, role, device):
        super().__init__(config, env, role, device)
        self.network = PPONetwork(env.observationShape, env.actionSpace)
        self.network.to(self.device)
        self.icm = ICMNetwork(env.observationShape, env.actionSpace)
        self.icm.to(self.device)
        self.rewardNormalizer = StdNormalizer()
        if role == Role.Trainer:
            self.network.buildOptimizer(self.config.learningRate)
            self.icm.buildOptimizer(self.config.learningRate)

    def getAgentHandler(self, env):
        return AgentHandler(self, env)

    def dump(self):
        data = {}
        data["network"] = self._getStateDict(self.network)
        data["icm"] = self._getStateDict(self.icm)
        data["rewardNormalizer"] = pickle.dumps(self.rewardNormalizer)
        return data

    def load(self, data):
        self.network.load_state_dict(data["network"])
        self.icm.load_state_dict(data["icm"])
        self.rewardNormalizer = pickle.loads(data["rewardNormalizer"])
    
    def preprocess(self, memory):
        with torch.no_grad():
            lastValue = 0
            lastMemory = memory[-1]
            if not lastMemory.next.info.done:
                lastState = KyleList([lastMemory.next.info]).toTensor(torch.float, self.device)
                hiddenState = KyleList([lastMemory.next.info.hiddenState]).toTensor(torch.float, self.device)
                lastValue, _ = self.network.getValue(lastState, hiddenState)
                lastValue = lastValue.item()
            
            returns = memory.select(lambda x: x.reward).toArray()
            # returns = self.rewardNormalizer.normalize(returns, update=True)
            
            # GAE (General Advantage Estimation)
            # Paper: https://arxiv.org/abs/1506.02438
            # Code: https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L55-L64
            gae = 0
            for i in reversed(range(len(memory))):
                transition = memory[i]
                reward = returns[i]
                detlas = reward + self.config.gamma * lastValue * (1 - transition.next.info.done) - transition.action.value
                gae = detlas + self.config.gamma * self.config.gaeCoeff * gae * (1 - transition.next.info.done)
                # from baseline
                # https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py#L65
                # transition.advantage = gae
                transition.reward = gae + transition.action.value
                lastValue = transition.action.value

    def learn(self, memory):
        self.network.train()
        self.icm.train()
        
        self.network.optimizer.zero_grad()
        self.icm.optimizer.zero_grad()
        batchSize = min(memory.size(), self.config.batchSize)
        n_miniBatch = memory.size() // batchSize
        totalLoss = 0

        # Normalize advantages
        # https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L139
        advantages = memory.select(lambda x: x.reward) - memory.select(lambda x: x.action.value)
        advantages = Function.normalize(advantages)

        for i in range(n_miniBatch):
            startIndex = i * batchSize
            minibatch = memory.get(startIndex, batchSize)

            # Get Tensors
            batch_states = minibatch.select(lambda x: x.info.state).toTensor(torch.float, self.device)
            
            batch_masks = minibatch.select(lambda x: x.info.mask).toTensor(torch.bool, self.device)
            batch_hiddenStates = minibatch.select(lambda x: x.info.hiddenState).toTensor(torch.float, self.device)

            # nextStates = minibatch.select(lambda x: x.nextState).toTensor(torch.float, self.device)
            batch_actions = minibatch.select(lambda x: x.action.index).toTensor(torch.long, self.device)
            batch_log_probs = minibatch.select(lambda x: x.action.log).toTensor(torch.float, self.device)
            batch_returns = minibatch.select(lambda x: x.reward).toTensor(torch.float, self.device)
            # batch_returns = returns.get(startIndex, batchSize).toTensor(torch.float, self.device)
            # batch_values = minibatch.select(lambda x: x.value).toTensor(torch.float, self.device)
            # print("hiddenStates:", batch_hiddenStates)
            
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
            probs, values, _ = self.network(batch_states, batch_hiddenStates, batch_masks)
            values = values.squeeze(-1)
            # print("returns:", batch_returns)
            # print("values:", values)
            # PPO2 - Confirm the samples aren't too far from pi.
            # porb1 / porb2 = exp(log(prob1) - log(prob2))
            dist = torch.distributions.Categorical(probs=probs)
            ratios = torch.exp(dist.log_prob(batch_actions) - batch_log_probs)
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
        nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)
        nn.utils.clip_grad.clip_grad_norm_(self.icm.parameters(), 0.5)

        self.network.optimizer.step()
        self.icm.optimizer.step()

        return totalLoss

class PPOAlgo(Algo[PPOConfig]):
    def __init__(self, config=PPOConfig()):
        super().__init__("PPO", config)

    def createHandler(self, env, role, device):
        return PPOHandler(self.config, env, role, device)
