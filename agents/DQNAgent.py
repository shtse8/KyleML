import os
import __main__
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
from utils.PredictionHandler import PredictionHandler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, output_size, name = "model"):
        super().__init__()
        self.name = name

        self.linear = nn.Linear(input_size, 256)
        # self.linear2 = nn.Linear(2048, 2048)
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        # x = F.relu(self.linear2(x))

        value = self.value(x)
        adv = self.adv(x)
        advAvg = adv.mean(1, keepdim=True)
        Q = value + adv - advAvg
        return Q

    def loss(self, input, target, weight):
        # weighted_mse_loss
        return (weight * (input - target) ** 2).mean()

class DQNAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("dqn", env, **kwargs)
        
        # Trainning
        self.update_target_every = kwargs.get('update_target_every', 1000)
        self.learning_rate = kwargs.get('learning_rate', .001)
        self.gamma = kwargs.get('gamma', 0.99)
        
        # Exploration
        self.epsilon_max = kwargs.get('epsilon_max', 1.00)
        self.epsilon_min = kwargs.get('epsilon_min', 0.10)
        self.epsilon_phase_size = kwargs.get('epsilon_phase_size', 0.2)
        self.epsilon = self.epsilon_max
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_trains * self.epsilon_phase_size))
        self.beta_increment_per_sampling = ((1 - 0.4) / self.target_trains)

        # Memory
        self.memory_size = kwargs.get('memory_size', 10000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = PrioritizedMemory(self.memory_size, self.beta_increment_per_sampling)
        self.stmemory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.network = Net(np.product(self.env.observationSpace), self.env.actionSpace, "model")
        self.network.to(self.device)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Target model
        self.network_target = Net(np.product(self.env.observationSpace), self.env.actionSpace, "target_model")
        self.network_target.to(self.device)
        
        self.target_update_counter = 0

        self.addModels(self.network)
        self.addModels(self.network_target)

    def beginPhrase(self):
        self.epsilon = self.epsilon_max
        self.stmemory.clear()
        self.ltmemory = PrioritizedMemory(self.memory_size, self.beta_increment_per_sampling)
        self.updateTarget()
        return super().beginPhrase()
    
    def commit(self, transition: Transition):
        if self.isTraining():
            # self.stmemory.add(transition)
            self.ltmemory.add(transition)
        super().commit(transition)
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def getPrediction(self, state):
        if self.isTraining() and np.random.uniform() < self.epsilon:
            prediction = np.random.rand(self.env.actionSpace)
        else:
            self.network.eval()
            with torch.no_grad():
                stateTensor = torch.FloatTensor([state.flatten()]).to(self.device)
                prediction = self.network(stateTensor).squeeze(0).cpu().detach().numpy()
        return prediction

    def getAction(self, prediction, mask = None):
        handler = PredictionHandler(prediction, mask)
        return handler.getBestAction()

    def endEpisode(self):
        if self.isTraining():
            self.learn()
            self.update_epsilon()
        
        super().endEpisode()
      
    def learn(self):
        self.network.train()
        
        idxs, batch, is_weights = self.ltmemory.sample(self.minibatch_size)
        
        # print(idxs, batch, is_weight)
        
        states = np.array([x.state.flatten() for x in batch])
        states = torch.FloatTensor(states).to(self.device)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        rewards = np.array([x.reward for x in batch])
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        dones = np.array([x.done for x in batch])
        dones = torch.FloatTensor(dones).to(self.device)
                
        nextStates = np.array([x.nextState.flatten() for x in batch])
        nextStates = torch.FloatTensor(nextStates).to(self.device)
        
        target_next_q_values = self.network_target(nextStates)
        
        results = self.network(torch.cat((states, nextStates), 0))
        q_values = results[0:len(states)]
        next_q_values = results[len(states):]
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.gather(1, next_q_values.argmax(1).unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        error = (q_value - expected_q_value).abs()

        self.ltmemory.batch_update(idxs, error.detach().numpy())
        
        is_weights = torch.FloatTensor(is_weights).to(self.device)
        loss = self.network.loss(q_value, expected_q_value, is_weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        # If the divergence of loss value is caused by gradient explode, you can clip the gradient. In Deepmind's 2015 DQN, the author clipped the gradient by limiting the value within [-1, 1]. 
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # In the other case, the author of Prioritized Experience Replay clip gradient by limiting the norm within 10.
        # nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        # Update target model counter every episode
        self.target_update_counter += 1
        
        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter >= self.update_target_every:
            self.updateTarget()
        
        # Stats
        self.report.trained(loss.item(), len(batch))
        
    def updateTarget(self):
        # print("Target is updated.")
        self.network_target.load_state_dict(self.network.state_dict())
        self.target_update_counter = 0
                