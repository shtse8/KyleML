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


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
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

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

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

class DQNAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Trainning
        self.weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + ".h5")
        self.update_target_every = kwargs.get('update_target_every', 1000)
        self.learning_rate = kwargs.get('learning_rate', .001)
        self.gamma = kwargs.get('gamma', 0.99)
        
        # Exploration
        self.epsilon_max = kwargs.get('epsilon_max', 1.00)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_phase_size = kwargs.get('epsilon_phase_size', 0.5)
        self.epsilon = self.epsilon_max
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_trains * self.epsilon_phase_size))
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 100000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = PrioritizedMemory(self.memory_size)
        self.stmemory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.model = Net(np.product(self.env.observationSpace), self.env.actionSpace)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Target model
        self.model_target = Net(np.product(self.env.observationSpace), self.env.actionSpace)
        
        self.target_update_counter = 0

    def beginPhrase(self):
        self.epsilon = self.epsilon_max
        self.stmemory.clear()
        self.ltmemory = PrioritizedMemory(self.memory_size)
        self.updateTarget()
        self.target_update_counter = 0
        return super().beginPhrase()
    
    def printSummary(self):
        print(self.model)

    def commit(self, transition: Transition):
        self.stmemory.add(transition)
        super().commit(transition)
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def getAction(self, state):
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
            
            _, _, error = self.prepareData(batch)
             
            for e, t in zip(error, batch):
                self.ltmemory.add(e.item(), t) 

            self.stmemory.clear()
            
            self.learn()
            self.update_epsilon()
        
        super().endEpisode()
      
    def learn(self):
        
        # loss = 0
        
        # batch_size = min(len(self.stmemory), self.minibatch_size)
        # batch = random.sample(self.stmemory, batch_size)
        idxs, batch, is_weights = self.ltmemory.sample(self.minibatch_size)
        # print(idxs, batch, is_weight)
        
        q_value, expected_q_value, error = self.prepareData(batch)
        self.ltmemory.batch_update(idxs, error.detach().numpy())
        
        is_weights = torch.tensor(is_weights, dtype=torch.float)
        loss = self.weighted_mse_loss(q_value, expected_q_value, is_weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target model counter every episode
        self.target_update_counter += 1
        
        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter >= self.update_target_every:
            self.updateTarget()
            
        self.lossHistory.append(loss.item())
    
    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).mean()

    def prepareData(self, batch):
        self.model.train()
        
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
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = target_next_q_values.gather(1, next_q_values.argmax(1).unsqueeze(1)).squeeze(1)
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
                