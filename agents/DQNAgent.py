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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 512)
        # self.linear2 = nn.Linear(2048, 2048)
        self.value = nn.Linear(512, 1)
        self.adv = nn.Linear(512, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        # x = F.relu(self.linear2(x))

        value = self.value(x)
        adv = self.adv(x)

        advAverage = adv.mean(1, keepdim=True)
        Q = value + adv - advAverage
        return Q

class DQNAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Trainning
        self.weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + ".h5")
        self.update_target_every = kwargs.get('update_target_every', 100)
        self.learning_rate = kwargs.get('learning_rate', .001)
        self.gamma = kwargs.get('gamma', 0.995)
        
        # Exploration
        self.epsilon_max = kwargs.get('epsilon_max', 1.00)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_phase_size = kwargs.get('epsilon_phase_size', 0.5)
        self.epsilon = self.epsilon_max
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_trains * self.epsilon_phase_size))
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 10000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 32)
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = PrioritizedMemory(self.memory_size)
        self.stmemory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.model = Net(np.product(self.env.observationSpace), self.env.actionSpace)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Target model
        self.model_target = Net(np.product(self.env.observationSpace), self.env.actionSpace)
        
        self.target_update_counter = 0

    def beginEpoch(self):
        self.epsilon = self.epsilon_max
        self.stmemory.clear()
        self.ltmemory = PrioritizedMemory(self.memory_size)
        self.updateTarget()
        self.target_update_counter = 0
        return super().beginEpoch()
        
    def printSummary(self):
        print(self.model)

    def commit(self, transition: Transition):
        self.stmemory.add(transition)
        super().commit(transition)
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_action(self, state):
        if self.isTraining() and np.random.uniform() < self.epsilon:
            action = random.randint(0, self.env.actionSpace - 1)
        else:
            self.model.eval()
            stateTensor = torch.tensor(state, dtype=torch.float).view(1, -1)
            prediction = self.model(stateTensor)
            action = prediction.argmax().item()
        return action
    
    def endEpisode(self):
        if self.isTraining():
            batch = self.stmemory.get()
            
            q_value, expected_q_value, error = self.prepareData(batch)
             
            for e, t in zip(error, batch):
                self.ltmemory.add(e.item(), t) 

            self.stmemory.clear()
            
            self.learn()
            self.update_epsilon()
        
        super().endEpisode()
      
    def learn(self):
        self.model.train()
        
        # loss = 0
        
        # batch_size = min(len(self.stmemory), self.minibatch_size)
        # batch = random.sample(self.stmemory, batch_size)
        idxs, batch, is_weights = self.ltmemory.sample(self.minibatch_size)
        # print(idxs, batch, is_weight)
        
        q_value, expected_q_value, error = self.prepareData(batch)
        
        # Loss with sample weights
        loss = (torch.tensor(is_weights, dtype=torch.float) * F.mse_loss(q_value, expected_q_value)).mean()
        
        self.ltmemory.batch_update(idxs, error.detach().numpy())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target model counter every episode
        self.target_update_counter += 1
        
        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter >= self.update_target_every:
            self.updateTarget()
            
        self.lossHistory.append(loss.item())

    def prepareData(self, batch):
        states = np.array([x.state for x in batch])
        states = torch.tensor(states, dtype=torch.float).view(states.shape[0], -1)
        
        actions = np.array([x.action for x in batch])
        actions = torch.tensor(actions, dtype=torch.long)
        
        rewards = np.array([x.reward for x in batch])
        rewards = torch.tensor(rewards, dtype=torch.float)
        
        dones = np.array([x.done for x in batch])
        dones = torch.tensor(dones, dtype=torch.float)
                
        nextStates = np.array([x.nextState for x in batch])
        nextStates = torch.tensor(nextStates, dtype=torch.float).view(nextStates.shape[0], -1)
        
        target_next_q_values = self.model_target(nextStates)
        
        results = self.model(torch.cat((states, nextStates), 0))
        q_values = results[0:len(states)]
        next_q_values = results[len(states):]
        # print(results, len(results), current_qs_list, len(current_qs_list), next_qs_list)
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        error = (q_value - expected_q_value).abs()
        
        return q_value, expected_q_value, error
    
    def save(self):
        try:
            torch.save(self.model.state_dict(), self.weights_path)
            # print("Saved Weights.")
        except:
            print("Failed to save.")
        
    def load(self):
        try:
            # self.model.load_weights(self.weights_path)
            self.model.load_state_dict(torch.load(self.weights_path))
            print("Weights loaded.")
        except:
            print("Failed to load.")
    
    def updateTarget(self):
        # print("Target is updated.")
        self.model_target.load_state_dict(self.model.state_dict())
        self.target_update_counter = 0
                