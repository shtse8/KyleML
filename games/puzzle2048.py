import numpy as np
import math

from utils.errors import InvalidAction
from .src.game2048 import game2048
from .Game import Game
import gym
# import gym_2048


class Puzzle2048(Game):
    def __init__(self):
        super().__init__()
        self.name = "game2048"
        self.size = 4
        self.game = game2048(self.size)
        self.observationSpace = (self.size, self.size)
        self.actionSpace = 4
        self.reward = 0
    
    def reset(self):
        self.game = game2048(self.size)
        return self.getState()
        
    def getState(self):
        state = np.zeros((self.size, self.size), dtype=int)
        for _, _, cell in self.game.grid.eachCell():
            if cell:
                state[cell.x][cell.y] = math.log2(cell.value)
                # state[cell.x][cell.y] = math.log2(cell.value)
        # print("score", self.game.score)
        # if self.game.score > 10000:
        #     print(state)
        return state
        
    def takeAction(self, action):
        score = self.game.score
        moved = self.game.move(action)
        if not moved:
            raise InvalidAction()
        self.reward = self.game.score - score
        # print("takeAction", self.reward)
        # self.game.update()
        return super().takeAction(action)
        
    def getDone(self):
        return self.game.isGameTerminated()
        
    def getReward(self):
        return self.reward
    
    # def render(self):
        # return self.game.render()