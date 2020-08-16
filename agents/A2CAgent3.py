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
from .Agent import Agent, Phrase, InvalidAction
from utils.PredictionHandler import PredictionHandler

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

class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, learingRate = 0.001, name = "default"):
        super(Network, self).__init__()
        self.name = name
        self.learningRate = learingRate

        hidden_nodes = 32
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden_nodes),
            nn.ReLU())
            
        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, n_outputs))
            
        # Define value head
        self.value = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1))

            
        self.optimizer = optim.Adam(self.parameters(), lr = self.learningRate)

    def predict(self, state):
        body_output = self.get_body_output(state)
        probs = F.softmax(self.policy(body_output), dim=-1)
        return probs, self.value(body_output)

    def get_body_output(self, state):
        return self.body(state)
    
    def critic(self, state):
        return self.value(self.get_body_output(state))

    # def get_action(self, state):
        # probs = self.predict(state)[0].detach().numpy()
        # action = np.random.choice(self.action_space, p=probs)
        # return action
    
    def get_log_probs(self, state):
        body_output = self.get_body_output(state)
        logprobs = F.log_softmax(self.policy(body_output), dim=-1)
        return logprobs    
        

class A2CAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.name="a2c"

        # Trainning
        self.learningRate = kwargs.get('learningRate', .001)
        self.gamma = kwargs.get('gamma', 0.9)
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 10000)
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.memory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.network = Network(
            np.product(self.env.observationSpace), 
            self.env.actionSpace, 
            learingRate=self.learningRate)
        
        self.n_commits = 0
        self.n_steps = 1000
        self.beta = 0.001
        self.zeta = 0.001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.addModels(self.network)
        
    def commit(self, transition: Transition):
        super().commit(transition)
        if self.isTraining():
            self.memory.add(transition)
            self.n_commits += 1
            if transition.done or self.n_commits % self.n_steps == 0:
                self.learn()
        
    def getPrediction(self, state):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).view(1, -1).to(self.device)
            prediction = self.network.predict(state)[0].squeeze(0)
            return prediction.cpu().detach().numpy()
        

    def getAction(self, prediction, mask = None):
        handler = PredictionHandler(prediction, mask)
        return handler.getRandomAction() if self.isTraining() else handler.getBestAction()
    
    def beginEpisode(self):
        self.memory.clear()
        self.n_commits = 0
        return super().beginEpisode()

    def getDiscountedRewards(self, rewards, gamma, finalReward):
        discountRewards = np.zeros_like(rewards).astype(float)
        runningReward = finalReward
        for i in reversed(range(len(rewards))):
            runningReward = runningReward * gamma + rewards[i]
            discountRewards[i] = runningReward
        return discountRewards
        
    def learn(self):
        self.network.train()
        
        
        batch = self.memory
        if len(batch) == 0:
            return

        states = np.array([x.state for x in batch])
        states = torch.FloatTensor(states).to(self.device).view(states.shape[0], -1)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        action_probs, values = self.network.predict(states)

        with torch.no_grad():
            rewards = np.array([x.reward for x in batch])
            finalReward = 0
            if not batch[-1].done:
                nextState = torch.FloatTensor(batch[-1].nextState).to(self.device).view(1, -1)
                finalReward = self.network.critic(nextState).item()
            discountRewards = self.getDiscountedRewards(rewards, self.gamma, finalReward)
            advantages = discountRewards - values.detach().numpy().flatten()

        discountRewards = torch.FloatTensor(discountRewards).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # print("calc_rewards", self.calc_rewards(states, actions, rewards, dones, nextStates))
        # discountRewards = torch.FloatTensor(discountRewards).to(self.device)
        # advantages = torch.FloatTensor(advantages).to(self.device)
        
        # final_reward = 0
        # state_values = self.network.predict(states)[1]
        # next_state_values = self.network.predict(next_states)[1]
        # discountRewards = torch.tensor(self.getDiscountedRewards(rewards, self.gamma, final_reward), dtype=torch.float)
        # print("discountRewards", discountRewards)
        # advantages = discountRewards - state_values
        log_probs = self.network.get_log_probs(states)
        log_prob_actions = advantages * log_probs.gather(1, actions.unsqueeze(1)).squeeze(1).sum()
        policy_loss = -log_prob_actions.mean()
        
        entropy_loss = -self.beta * (action_probs * log_probs).sum(dim=1).mean()
        
        value_loss = self.zeta * nn.MSELoss()(values.squeeze(-1), discountRewards)
        total_policy_loss = policy_loss - entropy_loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        self.network.optimizer.zero_grad()
        total_policy_loss.backward(retain_graph=True)
        value_loss.backward()
        # nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)
        self.network.optimizer.step()
        
        # Stats
        steps = len(batch)
        self.loss += total_loss.item() * steps
        self.steps += steps
        
        self.memory.clear()
        
    