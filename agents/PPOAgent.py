import sys
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

# def init_layer(m):
#     weight = m.weight.data
#     weight.normal_(0, 1)
#     weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
#     nn.init.constant_(m.bias.data, 0)
#     return m
    
class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, name="default"):
        super().__init__()
        self.name = name

        hidden_nodes = 64
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden_nodes),
            nn.Tanh())
            
        # Define policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, n_outputs),
            nn.Softmax(dim=-1))
            
        # Define value head
        self.value = nn.Sequential(
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, 1))

    def forward(self, state):
        body_output = self.get_body_output(state)
        return self.policy(body_output), self.value(body_output)

    def get_body_output(self, state):
        return self.body(state)
    
    def actor(self, state):
        body_output = self.get_body_output(state)
        return self.policy(body_output)

    def critic(self, state):
        return self.value(self.get_body_output(state))
        

class PPOAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("ppo", env, **kwargs)

        # Trainning
        self.learningRate: float = kwargs.get('learningRate', .001)
        self.gamma: float = kwargs.get('gamma', 0.99)

        # Memory
        self.memory_size: int = kwargs.get('memory_size', 10000)

        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.memory: SimpleMemory = SimpleMemory(self.memory_size)

        # Prediction model (the main Model)
        self.network: Network = Network(
            np.product(self.env.observationSpace),
            self.env.actionSpace)
        self.targetNetwork: Network = Network(
            np.product(self.env.observationSpace),
            self.env.actionSpace)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learningRate)
        self.updateTargetNetwork()
        
        self.network.to(self.device)
        self.addModels(self.network)
  
    def commit(self, transition: Transition):
        super().commit(transition)
        if self.isTraining():
            self.memory.add(transition)
            if transition.done:
                self.learn()

    def getPrediction(self, state):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).view(1, -1).to(self.device)
            prediction = self.targetNetwork.actor(state).squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, mask = None):
        handler = PredictionHandler(prediction, mask)
        return handler.getRandomAction() if self.isTraining() else handler.getBestAction()

    def beginEpisode(self) -> None:
        self.memory.clear()
        return super().beginEpisode()

    def getDiscountedRewards(self, rewards, gamma, finalReward):
        discountRewards = np.zeros_like(rewards).astype(float)
        runningReward = finalReward
        for i in reversed(range(len(rewards))):
            runningReward = runningReward * gamma + rewards[i]
            discountRewards[i] = runningReward
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
        
        predictions = np.array([x.prediction for x in batch])
        predictions = torch.FloatTensor(predictions).to(self.device)
        real_probs = torch.distributions.Categorical(probs=predictions).log_prob(actions).detach()

        # with torch.no_grad():
        rewards = np.array([x.reward for x in batch])
        finalReward = 0
        if not batch[-1].done:
            nextState = torch.FloatTensor(batch[-1].nextState).to(self.device).view(1, -1)
            finalReward = self.network.critic(nextState).item()
        targetValues = self.getDiscountedRewards(rewards, self.gamma, finalReward)
        targetValues = torch.FloatTensor(targetValues).to(self.device)
        
        eps_clip = 0.2
        for _ in range(4):
            action_probs, values = self.network(states)
            values = values.squeeze(1).detach()
            advantages = targetValues - values
            # advantages = Function.normalize(advantages)
            
            dist = torch.distributions.Categorical(probs=action_probs)
            # print(predictions)
            ratios = torch.exp(dist.log_prob(actions) - real_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist.entropy().mean()
            # value_loss = advantages.pow(2).mean()
            value_loss = F.mse_loss(values, targetValues)
            # print(value_loss, value_loss2)
            loss = actor_loss + 0.01 * entropy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            self.report.trained(loss.item(), len(batch))
            
        self.updateTargetNetwork()
        # Stats
        self.memory.clear()

    def updateTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.network.state_dict())
