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

        hidden_nodes = 128
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden_nodes * 2),
            nn.ReLU(),
            nn.Linear(hidden_nodes * 2, hidden_nodes),
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

    def forward(self, state, mask):
        output = self.body(state)
        
        policy = self.policy(output)
        if mask is not None:
            policy = policy * mask
        policy /= policy.sum(1).unsqueeze(1)
        return policy, self.value(output)

    def getPolicy(self, state, mask):
        output = self.body(state)
        output = self.policy(output)
        if mask is not None:
            output = output * mask
        output /= output.sum(1).unsqueeze(1)
        return output

    def getValue(self, state):
        output = self.body(state)
        return self.value(output)
        

class PPOAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__("ppo", env, **kwargs)

        # Trainning
        self.learningRate: float = kwargs.get('learningRate', .0001)
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
        # self.schedular = schedular.StepLR(self.optimizer, step_size=1, gamma=0.999)
        
        self.network.to(self.device)
        self.addModels(self.network)
  
    def commit(self, transition: Transition):
        super().commit(transition)
        if self.isTraining():
            self.memory.add(transition)
            if transition.done:
                self.learn()

    def getPrediction(self, state, mask=None):
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor([state.flatten()]).to(self.device)
            mask = torch.FloatTensor([mask.flatten()]).to(self.device)
            prediction = self.network.getPolicy(state, mask).squeeze(0)
            return prediction.cpu().detach().numpy()

    def getAction(self, prediction, mask=None):
        handler = PredictionHandler(prediction, mask)
        if self.isTraining():
            return handler.getRandomAction()
        else:
            return handler.getBestAction()

    def beginEpisode(self) -> bool:
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
        gae = 0
        lastValue = 0
        for i in reversed(range(len(rewards))):
            value = values[i].item()
            detlas = rewards[i] + self.gamma * lastValue * (1 - dones[i]) - value
            gae = detlas + self.gamma * 0.95 * gae * (1 - dones[i])
            advantages[i] = gae
            lastValue = value
        return advantages

    def learn(self) -> None:
        self.network.train()

        batch = self.memory
        if len(batch) == 0:
            return

        states = np.array([x.state.flatten() for x in batch])
        states = torch.FloatTensor(states).to(self.device)
        
        actions = np.array([x.action for x in batch])
        actions = torch.LongTensor(actions).to(self.device)
        
        predictions = np.array([x.prediction for x in batch])
        predictions = torch.FloatTensor(predictions).to(self.device)

        dones = np.array([x.done for x in batch])
        # dones = torch.BoolTensor(dones).to(self.device)

        real_probs = torch.distributions.Categorical(probs=predictions).log_prob(actions)

        rewards = np.array([x.reward for x in batch])

        targetValues = self.getDiscountedRewards(rewards, dones)
        targetValues = torch.FloatTensor(targetValues).to(self.device)
        # targetValues = Function.normalize(targetValues)
        # print(rewards, targetValues)
        # with torch.no_grad():
        
        masks = np.array([self.env.getActionMask(x.state) for x in batch])
        masks = torch.FloatTensor(masks).to(self.device)
        # print(masks)

        eps_clip = 0.2
        for _ in range(4):
            action_probs, values = self.network(states, masks)
            values = values.squeeze(1)

            advantages = self.getAdvantages(rewards, dones, values)
            advantages = torch.FloatTensor(advantages).to(self.device)
            advantages = Function.normalize(advantages)
            # print(advantages)

            dist = torch.distributions.Categorical(probs=action_probs)
            # print(predictions)
            ratios = torch.exp(dist.log_prob(actions) - real_probs)  # porb1 / porb2 = exp(log(prob1) - log(prob2))
            surr1 = ratios * advantages
            surr2 = ratios.clamp(1 - eps_clip, 1 + eps_clip) * advantages

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
            self.report.trained(loss.item(), len(batch))
            
        # Stats
        self.memory.clear()

