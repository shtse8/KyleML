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
import copy

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


class AgentHandler:
    def __init__(self, handler, env):
        self.handler = handler
        self.env = env

    def reset(self):
        pass

    def reportStep(self, action):
        self.handler.reportStep(action)

    def getInfo(self):
        info = self.env.getInfo()
        return info

    def getProb(self):
        return self.handler.getProb(self.env)

    def getAction(self, isTraining: bool):
        acts, probs = self.handler.mcts.get_move_probs(self.env, temp=1)
        if isTraining:
            noise_probs = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            index = np.random.choice(len(probs), p=noise_probs)
        else:
            index = np.argmax(probs)
        prediction = np.zeros(self.env.game.actionSpace)
        prediction[list(acts)] = probs
        # state = self.env.getState()
        # mask = self.env.getMask(state)
        # print(state[0] + state[1] * 2, acts, probs, index, "\n")
        self.handler.mcts.update_with_move(acts[index])
        return Action(
            index=acts[index],
            probs=prediction)


class AlphaZeroHandler(AlgoHandler):
    def __init__(self, config, env, role, device):
        super().__init__(config, env, role, device)
        self.network = AlphaZeroNetwork(env.observationShape, env.actionSpace)
        self.network.to(self.device)
        self.mcts = MCTS(self.getProb, n_playout=self.config.simulations)
        if role == Role.Trainer:
            self.network.buildOptimizer(self.config.learningRate)

    def getProb(self, env):
        self.network.eval()
        with torch.no_grad():
            state = env.getState()
            mask = env.getMask()
            stateTensor = torch.tensor([state], dtype=torch.float, device=self.device)
            maskTensor = torch.tensor([mask], dtype=torch.bool, device=self.device)
            prob, value = self.network(stateTensor, maskTensor)
            return prob.squeeze(0).cpu().detach().numpy(), \
                value.item()

    def getAgentHandler(self, env):
        return AgentHandler(self, env)

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
        
    def preprocess(self, memory):
        with torch.no_grad():
            lastValue = 0
            for i in reversed(range(len(memory))):
                transition = memory[i]
                transition.reward = transition.reward + lastValue
                lastValue = transition.reward

    def learn(self, memory):
        self.network.train()
        batchSize = min(memory.size(), self.config.batchSize)
        n_miniBatch = memory.size() // batchSize

        self.network.optimizer.zero_grad()
        totalLoss = 0
        for i in range(n_miniBatch):
            startIndex = i * batchSize
            minibatch = memory.get(startIndex, batchSize)

            # Get Tensors
            states = minibatch.select(lambda x: x.info.state).toTensor(torch.float, self.device)
            masks = minibatch.select(lambda x: x.info.mask).toTensor(torch.bool, self.device)
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

        return [totalLoss]
        
class AlphaZeroAlgo(Algo[AlphaZeroConfig]):
    def __init__(self, config=AlphaZeroConfig()):
        super().__init__("AlphaZero", config)

    def createHandler(self, env, role, device):
        return AlphaZeroHandler(self.config, env, role, device)



def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

#定义节点
class TreeNode(object):
    """A node in the MCTS tree.
    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, player_id = 0):
        self._parent = parent   #父节点
        self._children = {}     # 子节点，是一个字典：字典的key是动作，item是子节点。子节点包括了描述这个动作的概率，Q等
        self._n_visits = 0      # 记录这个节点被访问次数
        self._Q = 0             #这个节点的价值
        self._u = 0             #用于计算UCB上限。在select的时候，用的是Q+U的最大值。
        self._P = prior_p       #动作对应的概率
        self.player_id = player_id

    def expand(self, action_priors, mask, player_id):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        展开：当一个节点是叶子节点的时候，需要被展开。
            输入action_priors：包含action和对应的概率
            判断这个动作是否在_children的字典中。如果不在，增加这个动作，并增加对应的节点，把概率写在节点中。
        """
        for action, prob in enumerate(action_priors):
            if mask[action] and action not in self._children:
                self._children[action] = TreeNode(self, prob, player_id)

    def select(self, c_puct, player_id):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        选择：选择UCB最大的值：UCB = Q(s,a) + U(s,a)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct, player_id))

    def update(self, value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        做一次模拟，把返回的leaf_value去修改Q
        1._n_visits增加
        2.leaf_value和原来的Q，用_n_visits平均一下。1.0是学习率
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (value - self._Q) / self._n_visits
        # self._Q = 1.0 * value / self._n_visits

    def update_recursive(self, value, player_id):
        """Like a call to update(), but applied recursively for all ancestors.
        用leaf_value反向更新祖先节点。
        因为整棵树，是双方轮流下子的。所以对于一个state update是正的，那么这个state前后updata的数值就是负的。
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(value, player_id)
        if player_id != self.player_id:
            value *= -1
        self.update(value)

    def get_value(self, c_puct, player_id):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        UCB = Q(s,a) + U(s,a)
        """
        
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        value = self._Q + self._u
        if self.player_id != player_id:
            value *= -1
        return value

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=1, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0) #初始化根节点
        self._policy = policy_value_fn #用于生成子节点action-prob对
        self._c_puct = c_puct  #一个常数，好像没啥用
        self._n_playout = n_playout   #模拟多少次走一步

    #进行一次模拟_root就代表传入state
    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        
        node = self._root
        #直到去到叶子节点
        while not node.is_leaf():
            # Greedily select next move.
            #找出UCB最大的动作，并执行。
            # pred_node = node
            action, node = node.select(self._c_puct, env.playerId)
            # print(action, [(a, v.get_value(self._c_puct, env.playerId)) for a, v in pred_node._children.items()])
            # print(env.getState(), action)
            env.step(action)
            env = env.getNext()
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        # 我们评估这个叶子节点的Q，和他的action-probs
        # 如果还没有结束，那么就扩展这棵树。action-probs放进子节点。
        
        
        # Check for end of game.
        if not env.isDone():
            prob, value = self._policy(env)
            mask = env.getMask()
            node.expand(prob, mask, env.playerId)
        #如果结束了。
        # 如果平局，就设置成leaf_value = 0
        # 否则: 如果胜利者是当前的，那么leaf_value = 1， 否则leaf_value = -1
        else:
            # for end state，return the "true" leaf_value
            value = env.getDoneReward()
        # extrinsic reward
        value += env.getReward()
        
        # print("=====", value)
        # 向上更新祖先节点。
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(value, env.playerId)


    def get_move_probs(self, env, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        #每一步进行_n_playout次模拟
        #每次都把现在的state复制出来。进行模拟，一直到游戏结束
        #把得到的leaf_value更新到每个状态。同时更新被访问次数。
        #最后我们会得到一颗模拟出来各种结果的树，我们需要的就是这个树。
        # tic = time.perf_counter()
        for _ in range(self._n_playout):
            #关于copy.deepcopy(state)
            #https://blog.csdn.net/u010712012/article/details/79754132
            env_copy = copy.deepcopy(env)
            self._playout(env_copy)
        # print(time.perf_counter() - tic)
        # calc the move probabilities based on visit counts at the root node
        # _root._children.items()访问根节点的_children，就是访问当前状态下，各个动作和对应的节点。
        # 取出节点和被访问次数
        # 然后一轮运算后，根据访问次数，获得act 和对应的act_probs
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        # print(visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        #下棋后，检查这move是否在这个树的子节点中。如果在就把根节点移动到这个节点。
        #否则新建一个节点。
        #这棵树会一直维护，直到一次游戏结束。
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        # 输入-1，重置整棵树
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
