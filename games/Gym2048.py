import numpy as np

from operator import add
import gym
import gym_2048
from .Game import Game

class Gym2048(Game):
    def __init__(self):
        self.name = "2048-v0"
        self.game = gym.make(self.name)
        self.game._max_episode_steps = 10000
        # print(self.game.observation_space.shape)
        # print(self.game.action_space.n)
        self.observationSpace = self.game.observation_space.shape
        self.actionSpace = self.game.action_space.n
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
        # self.game.render()
        self.state, self.reward, self.done, _ = self.game.step(action)
        # print(self.state)
        return self.state, self.reward, self.done
        
    def getDone(self):
        return self.done
        
    def getReward(self):
        return self.reward
    
    