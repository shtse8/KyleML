import gym
import numpy as np

from .Game import Game


class GymGame(Game):
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.name = "gym-" + id
        self.game = gym.make(id)
        self.game._max_episode_steps = 10000

        shape = self.game.observation_space.shape
        if id == "Blackjack-v0":
            shape = (len(self.game.observation_space),)
        if len(shape) == 3:
            shape = (shape[2], shape[0], shape[1])
        self.observationShape = shape
        
        self.actionSpace = self.game.action_space.n
        self.state = None
        self.done = False
        self.reward = 0

    def _processState(self, state):
        if len(self.observationShape) == 3:
            # For pytorch
            state = np.einsum('ijk->kij', state)
        return state

    def reset(self):
        self.state = self.game.reset()
        self.state = self._processState(self.state)
        self.reward = 0
        self.done = False
        return self.state
        
    def getState(self):
        return self.state
        
    def takeAction(self, action):
        self.state, self.reward, self.done, info = self.game.step(action)
        self.state = self._processState(self.state)
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
    
    