import numpy as np
import math

from .src.puzzle2048 import Board as GameSrc, Direction
from .Game import Game

class Puzzle2048(Game):
    def __init__(self):
        self.name = "Snake"
        self.game = GameSrc()
        self.observationSpace = (4, 4)
        self.actionSpace = 4
        self.reward = 0
    
    def reset(self):
        self.game = GameSrc()
        # self.game.food.x = self.game.player.x + 2
        # self.game.food.y = self.game.player.y
        return self.getState()
        
    def getState(self):
        state = np.zeros((4, 4))
        for block in self.game.blocks:
            state[block.coordinate_y][block.coordinate_x] = math.log2(block.score)
        
        return state
        
    def takeAction(self, action):
        action_map = [Direction.Up, Direction.Down, Direction.Left, Direction.Right]
        score = self.game.score
        self.game.slide(action_map[action])
        self.reward = self.game.score - score
        return self.getState(), self.getReward(), self.getDone()
        
    def getDone(self):
        return self.game.is_end
        
    def getReward(self):
        return self.reward
    
    