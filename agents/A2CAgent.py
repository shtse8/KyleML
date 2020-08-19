import numpy as np
from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from .Agent import Agent
from utils.PredictionHandler import PredictionHandler
import utils.Function as Function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, name="default"):
        super().__init__()
        self.name = name

        hidden_nodes = 128
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden_nodes),
            nn.ReLU())
            
        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, n_outputs),
            nn.Softmax(dim=-1))
            
        # Define value head
        self.value = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, 1))

    def forward(self, state):
        output = self.body(state)
        return self.policy(output), self.value(output)

    def getPolicy(self, state):
        output = self.body(state)
        return self.policy(output)

    def getValue(self, state):
        output = self.body(state)
        return self.value(output)
        

class A2CAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("a2c", env, **kwargs)

        # Trainning
        self.learningRate: float = kwargs.get('learningRate', .001)
        self.gamma: float = kwargs.get('gamma', 0.9)

        # Memory
        self.memory_size: int = kwargs.get('memory_size', 10000)

        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.memory: SimpleMemory = SimpleMemory(self.memory_size)

        # Prediction model (the main Model)
        self.network: Network = Network(
            np.product(self.env.observationSpace),
            self.env.actionSpace)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learningRate)

        self.n_steps: int = 50

        self.network.to(self.device)
        self.addModels(self.network)
  
    def commit(self, transition: Transition):
        super().commit(transition)
        if self.isTraining():
            self.memory.add(transition)
            if transition.done:  # or self.steps % self.n_steps == 0:
                self.learn()

    def getPrediction(self, state):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).view(1, -1).to(self.device)
            prediction = self.network.getPolicy(state).squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, mask = None):
        handler = PredictionHandler(prediction, mask)
        return handler.getRandomAction() if self.isTraining() else handler.getBestAction()

    def beginEpisode(self) -> None:
        self.memory.clear()
        return super().beginEpisode()

    # Discounted Rewards (N-steps)
    def getDiscountedRewards(self, rewards, dones):
        discountedRewards = np.zeros_like(rewards).astype(float)
        runningDiscountedRewards = 0
        for i in reversed(range(len(rewards))):
            runningDiscountedRewards = rewards[i] + self.gamma * runningDiscountedRewards * (1 - dones[i])
            discountedRewards[i] = runningDiscountedRewards
        return discountedRewards

    def getAdvantages(self, rewards, dones, values):
        advantages = np.zeros_like(rewards).astype(float)
        runningReward = 0
        runningValue = 0
        for i in reversed(range(len(rewards))):
            detlas = rewards[i] + self.gamma * runningValue * (1 - dones[i]) - values[i]
            runningValue = values[i]
            runningReward = detlas + self.gamma * 0.95 * runningReward * (1 - dones[i])
            advantages[i] = runningReward
        return advantages

    def getDiscountedRewards2(self, rewards, gamma, finalReward):
        discountRewards = np.zeros_like(rewards).astype(float)
        runningReward = finalReward
        for i in reversed(range(len(rewards))):
            runningReward = runningReward * gamma + rewards[i]
            discountRewards[i] = runningReward

        discountRewards = (discountRewards - discountRewards.mean()) / (discountRewards.std() + 1e-5)
        return discountRewards

    def learn(self) -> None:
        self.network.train()

        batch = self.memory
        if len(batch) == 0:
            return

        states = np.array([x.state for x in batch])
        states = torch.FloatTensor(states).to(self.device).view(states.shape[0], -1)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        action_probs, values = self.network(states)
        values = values.squeeze(1)

        dones = np.array([x.done for x in batch])
        rewards = np.array([x.reward for x in batch])
        
        targetValues = self.getDiscountedRewards(rewards, dones)
        targetValues = torch.FloatTensor(targetValues).to(self.device)

        advantages = self.getAdvantages(rewards, dones, values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = Function.normalize(advantages)
        
        # finalReward = 0
        # if not batch[-1].done:
        #     nextState = torch.FloatTensor(batch[-1].nextState).to(self.device).view(1, -1)
        #     finalReward = self.network.critic(nextState).item()
        # targetValues = self.getDiscountedRewards2(rewards, self.gamma, finalReward)
        # targetValues = torch.FloatTensor(targetValues).to(self.device)
        # advantages = targetValues - values
        
        dist = torch.distributions.Categorical(probs=action_probs)
        policy_loss = -(dist.log_prob(actions) * advantages.detach()).mean()
        entropy_loss = -dist.entropy().mean()  # Maximize Entropy Loss
        value_loss = F.mse_loss(values, targetValues)  # Minimize Value Loss
        # value_loss = nn.MSELoss()(values, discountRewards)
        
        loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        # Stats
        self.report.trained(loss.item(), len(batch))
        self.memory.clear()
