import __main__
import collections
import math
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from memories.PrioritizedMemory import PrioritizedMemory
from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from utils.PredictionHandler import PredictionHandler
from .Agent import Agent


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, output_size, name="model"):
        super().__init__()
        self.name = name
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.body2 = nn.Sequential(
            nn.Linear(64 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.body2(x)
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
        self.update_target_every = 1000
        self.learning_rate = 0.001
        self.gamma = 0.99
        
        # Exploration
        self.epsilon_max = 0.2
        self.epsilon_min = 0
        self.epsilon_phase_size = 0.2
        self.epsilon = self.epsilon_max

        # Memory
        self.memory_size = 10000
        self.epsilon_decay = 0
        self.beta_increment_per_sampling = 0

        # Mini Batch
        self.minibatch_size = 64
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = PrioritizedMemory(self.memory_size, self.beta_increment_per_sampling)
        self.stmemory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.network = Net(np.product(self.env.observationShape), self.env.actionSpace, "model")
        self.network.to(self.device)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Target model
        self.network_target = Net(np.product(self.env.observationShape), self.env.actionSpace, "target_model")
        self.network_target.to(self.device)
        
        self.target_update_counter = 0

        self.addModels(self.network)
        self.addModels(self.network_target)

    def beginEpoch(self):
        self.epsilon = self.epsilon_max
        self.stmemory.clear()
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_episodes * self.epsilon_phase_size))
        self.beta_increment_per_sampling = ((1 - 0.4) / self.target_episodes)
        self.ltmemory = PrioritizedMemory(self.memory_size, self.beta_increment_per_sampling)
        self.updateTarget()
        return super().beginEpoch()
    
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
                stateTensor = torch.FloatTensor([[state]]).to(self.device)
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
        
        states = np.array([[x.state] for x in batch])
        states = torch.FloatTensor(states).to(self.device)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        rewards = np.array([x.reward for x in batch])
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        dones = np.array([x.done for x in batch])
        dones = torch.FloatTensor(dones).to(self.device)
                
        nextStates = np.array([[x.nextState] for x in batch])
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
        nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 1)
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
                