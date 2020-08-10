import numpy as np

from operator import add
import gym
from .Game import Game

class CartPole(Game):
    def __init__(self):
        self.name = "CartPole-v0"
        self.game = gym.make('CartPole-v0')
        self.game._max_episode_steps = 10000
        self.observationSpace = 4
        self.actionSpace = 2
        self.state = None
        self.done = False
        self.reward = 0
    
    def reset(self):
        self.state = self.game.reset()
        self.reward = 0
        self.done = False
        return self.state
        
    def getState(self):
        return self.state
        
    def takeAction(self, action):
        self.state, self.reward, self.done, _ = self.game.step(action)
        return self.state, self.reward, self.done
        
    def getDone(self):
        return self.done
        
    def getReward(self):
        return self.reward
    
    