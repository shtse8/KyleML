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
from torch.autograd import Variable

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class ActorNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, name = "actor"):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(s_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,a_dim)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        # out = F.log_softmax(self.fc3(out))
        out = F.softmax(self.fc3(out), dim=1)
        return out
        
class CriticNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, name = "critic"):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(s_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,a_dim)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        
    # def loss_func(self, s, a, v_t):
        # self.train()
        # logits, values = self.forward(s)
        # td = v_t - values
        # c_loss = td.pow(2)
        
        # probs = F.softmax(logits, dim=1)
        # m = self.distribution(probs)
        # exp_v = m.log_prob(a) * td.detach().squeeze()
        # a_loss = -exp_v
        # total_loss = (c_loss + a_loss).mean()
        # return total_loss

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
        self.actor = ActorNetwork(np.product(self.env.observationSpace), self.env.actionSpace)
        self.critic = CriticNetwork(np.product(self.env.observationSpace), 1)
        
        # self.model.cuda()
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.actorOptimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.criticOptimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        self.addModels(self.actor)
        self.addModels(self.critic)

    def beginPhrase(self):
        self.memory.clear()
        return super().beginPhrase()

    def commit(self, transition: Transition):
        super().commit(transition)
        self.memory.add(transition)
        
    def getPrediction(self, state):
        self.actor.eval()
        
        stateTensor = torch.tensor(state, dtype=torch.float).view(1, -1)
        log_softmax_action = self.actor(stateTensor)
        prediction = torch.exp(log_softmax_action).squeeze(0)
        return prediction.detach().numpy()
        

    def getAction(self, prediction, actionMask = None):
        if actionMask is not None:
            prediction = self.applyMask(prediction, actionMask)
        if self.isTraining():
            predictionSum = np.sum(prediction)
            prediction /= predictionSum
            action = np.random.choice(self.env.actionSpace, p=prediction)
        else:
            action = prediction.argmax()
        return action
    
    def endEpisode(self):
        if self.isTraining():
            self.learn()
        super().endEpisode()
      
    def discount_reward(self, r, gamma, final_r):
        discounted_r = np.zeros_like(r)
        running_add = final_r
        for t in reversed(range(0, len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
        
    def learn(self):
        self.actor.train()
        self.critic.train()
        
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
        
        final_r = 0
        # train actor network
        self.actorOptimizer.zero_grad()
        log_softmax_actions = self.actor(states)
        vs = self.critic(states).detach()
        # print("critic", vs)
        # calculate qs
        qs = torch.tensor(self.discount_reward(rewards, 0.99, final_r), dtype=torch.float)
        # print("qs", qs)
        advantages = qs - vs
        q_value = log_softmax_actions.gather(1, actions.unsqueeze(1)).squeeze(1)
        actorLoss = -torch.mean(q_value.sum() * advantages)
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actorOptimizer.step()
        
        
        # train value network
        self.criticOptimizer.zero_grad()
        target_values = qs.unsqueeze(1)
        values = self.critic(states)
        criterion = nn.MSELoss()
        criticLoss = criterion(values, target_values)
        criticLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),0.5)
        self.criticOptimizer.step()
        
        
        self.memory.clear()
            
        self.total_loss += actorLoss.item()
        self.total_loss += criticLoss.item()
    