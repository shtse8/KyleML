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
import torch.optim.lr_scheduler as schedular

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
            nn.Conv2d(n_inputs[0], 32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * n_inputs[1] * n_inputs[2], hidden_nodes),
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
        

class PPOAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("ppo", env, **kwargs)

        # Trainning
        self.learningRate: float = kwargs.get('learningRate', .0001)
        self.gamma: float = kwargs.get('gamma', 0.9)
        self.eps_clip = 0.2
        self.updateEpoches = 4

        # Prediction model (the main Model)
        self.network: Network = Network(
            self.env.observationShape,
            self.env.actionSpace)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learningRate)
        # self.schedular = schedular.StepLR(self.optimizer, step_size=1, gamma=0.999)
        
        self.network.to(self.device)
        self.addModels(self.network)
 
    def getPrediction(self, state):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor([state]).to(self.device)
            prediction = self.network.getPolicy(state).squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, mask=None):
        handler = PredictionHandler(prediction, mask)
        if self.isTraining():
            return handler.getRandomAction()
        else:
            return handler.getBestAction()

    # Discounted Rewards (N-steps)
    def getDiscountedRewards(self, rewards, dones, lastValue=0):
        discountedRewards = np.zeros_like(rewards).astype(float)
        for i in reversed(range(len(rewards))):
            lastValue = rewards[i] + self.gamma * lastValue * (1 - dones[i])
            discountedRewards[i] = lastValue
        return discountedRewards

    def getAdvantages(self, rewards, dones, values, lastValue=0):
        advantages = np.zeros_like(rewards).astype(float)
        gae = 0
        for i in reversed(range(len(rewards))):
            value = values[i].item()
            detlas = rewards[i] + self.gamma * lastValue * (1 - dones[i]) - value
            gae = detlas + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages[i] = gae
            lastValue = value
        return advantages

    def learn(self, batch) -> None:
        self.network.train()
        batchSize = len(batch) // self.updateEpoches
        for i in range(self.updateEpoches):
            startIndex = i * batchSize
            endIndex = min(startIndex + batchSize, len(batch) - 1)
            minibatch = batch[startIndex:endIndex]
            
            states = np.array([x.state for x in minibatch])
            states = torch.FloatTensor(states).to(self.device)
            
            actions = np.array([x.action for x in minibatch])
            actions = torch.LongTensor(actions).to(self.device)
            
            predictions = np.array([x.prediction for x in minibatch])
            predictions = torch.FloatTensor(predictions).to(self.device)

            dones = np.array([x.done for x in minibatch])
            # dones = torch.BoolTensor(dones).to(self.device)

            rewards = np.array([x.reward for x in minibatch])
            nextStates = np.array([x.nextState for x in minibatch])
            real_probs = torch.distributions.Categorical(probs=predictions).log_prob(actions)

            lastState = torch.FloatTensor([nextStates[-1]]).to(self.device)

            lastValue = 0 if dones[-1] else self.network.getValue(lastState).item()
            targetValues = self.getDiscountedRewards(rewards, dones, lastValue)
            targetValues = torch.FloatTensor(targetValues).to(self.device)
            
            action_probs, values = self.network(states)
            values = values.squeeze(1)

            advantages = self.getAdvantages(rewards, dones, values, lastValue)
            advantages = torch.FloatTensor(advantages).to(self.device)
            advantages = Function.normalize(advantages)

            dist = torch.distributions.Categorical(probs=action_probs)
            ratios = torch.exp(dist.log_prob(actions) - real_probs)  # porb1 / porb2 = exp(log(prob1) - log(prob2))
            surr1 = ratios * advantages
            surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()  # Maximize Policy Loss
            entropy_loss = -dist.entropy().mean()  # Maximize Entropy Loss
            value_loss = F.mse_loss(values, targetValues)  # Minimize Value Loss
            # print(policy_loss, entropy_loss, value_loss)
            loss = policy_loss + 0.01 * entropy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            # Chip grad with norm
            nn.utils.clip_grad.clip_grad_norm_(self.network.parameters(), 0.5)

            self.optimizer.step()

            # Report
            self.report.trained(loss.item(), len(minibatch))
            

