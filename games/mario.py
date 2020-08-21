import numpy as np
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from .Game import Game


class Mario(Game):
    def __init__(self):
        super().__init__()
        self.name = "SuperMarioBros"
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.game = env
        self.game._max_episode_steps = 10000
        shape = self.game.observation_space.shape
        if len(shape) == 3:
            shape = (shape[2], shape[0], shape[1])
        self.observationShape = shape
        self.actionSpace = self.game.action_space.n
        self.state = None
        self.done = False
        self.reward = 0
    
    def reset(self):
        self.state = self.game.reset()
        self.state = np.einsum('ijk->kij', self.state)
        self.reward = 0
        self.done = False
        return self.state
        
    def getState(self):
        return self.state
        
    def takeAction(self, action):
        self.state, self.reward, self.done, _ = self.game.step(action)
        self.state = np.einsum('ijk->kij', self.state)
        return self.getState(), self.getReward(), self.getDone()
        
    def getDone(self):
        return self.done
        
    def getReward(self):
        return self.reward
    
    