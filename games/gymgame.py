import numpy as np
import gym
from .Game import Game


class GymGame(Game):
    def __init__(self, name):
        super().__init__()
        self.name = "gym-" + name
        self.game = gym.make(name)
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
        return super().takeAction(action)
        
    def render(self) -> None:
        if self.rendered:
            return
        self.rendered = True
        self.game.render()

    def update(self):
        if self.rendered:
            self.game.render()
        
    def getDone(self):
        return self.done
        
    def getReward(self):
        return self.reward
    
    