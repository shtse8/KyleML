import os
import random
import numpy as np
import sys
import collections
import math
import time
from memories.PrioritizedMemory import PrioritizedMemory
from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from .Agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class Net(nn.Module):
    def __init__(self, s_dim, a_dim, name = "model"):
        super(Net, self).__init__()
        self.name = name
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)

        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical
        
    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class A2CAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.name="a2c"

        # Trainning
        self.learning_rate = kwargs.get('learning_rate', .001)
        self.gamma = kwargs.get('gamma', 0.99)
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 100000)
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.memory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.model = Net(np.product(self.env.observationSpace), self.env.actionSpace)
        # self.model.cuda()
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.addModels(self.model)

    def beginPhrase(self):
        self.memory.clear()
        return super().beginPhrase()

    def commit(self, transition: Transition):
        super().commit(transition)
        self.memory.add(transition)
        
    def getPrediction(self, state):
        self.model.eval()
        stateTensor = torch.tensor(state, dtype=torch.float).view(1, -1)
        # stateTensor = stateTensor.cuda()
        logits, _ = self.model(stateTensor)
        # logits = logits.detach().cpu()
        prediction = F.softmax(logits, dim=1).data
        return prediction

    def getAction(self, prediction, actionMask = None):
        if actionMask is not None:
            prediction *= actionMask
        if self.isTraining():
            action = self.model.distribution(prediction).sample().item()
        else:
            action = prediction.argmax()
        return action
    
    def endEpisode(self):
        if self.isTraining():
            self.learn()
        super().endEpisode()
      
    # def wrap(np_array, dtype=np.float32):
    #     if np_array.dtype != dtype:
    #         np_array = np_array.astype(dtype)
    #     return torch.from_numpy(np_array)

    def learn(self):
        self.model.train()
        
        batch = self.memory
        
        states = np.array([x.state for x in batch])
        states = torch.tensor(states, dtype=torch.float).view(states.shape[0], -1) #.cuda()
        
        actions = np.array([x.action for x in batch])
        actions = torch.tensor(actions, dtype=torch.long) #.cuda()
        
        rewards = np.array([x.reward for x in batch])
        # rewards = torch.tensor(rewards, dtype=torch.float)
        
        # dones = np.array([x.done for x in batch])
        # dones = torch.tensor(dones, dtype=torch.float)
                
        # nextStates = np.array([x.nextState for x in batch])
        # nextStates = torch.tensor(nextStates, dtype=torch.float).view(nextStates.shape[0], -1)
        

        # if done:
        #     v_s_ = 0.               # terminal
        # else:
        #     v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
        v_s_ = 0
        buffer_v_target = []
        # print(rewards, rewards[::-1])
        for r in rewards[::-1]:    # reverse buffer r
            v_s_ = r + self.gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        buffer_v_target = torch.tensor(buffer_v_target, dtype=torch.float) #.cuda()
        # print(buffer_v_target)
        
        
        loss = self.model.loss_func(states, actions, buffer_v_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        self.memory.clear()
            
        self.lossHistory.append(loss.item())
    