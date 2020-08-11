import numpy as np
import math

from .src.game2048 import game2048
from .Game import Game
import gym
# import gym_2048

class Puzzle2048(Game):
    def __init__(self):
        self.name = "game2048"
        self.size = 4
        self.game = game2048(self.size)
        self.observationSpace = (self.size, self.size)
        self.actionSpace = 4
        self.reward = 0
    
    def reset(self):
        self.game = game2048(4)
        # self.game.food.x = self.game.player.x + 2
        # self.game.food.y = self.game.player.y
        return self.getState()
        
    def getState(self):
        return self.game.game_state
        
    def takeAction(self, action):
        action_map = ["up", "down", "left", "right",]
        score = self.game.get_score()
        self.game.swipe(action_map[action])
        self.reward = self.game.get_score() - score
        # self.game.update()
        return self.getState(), self.getReward(), self.getDone()
        
    def getDone(self):
        return self.game.check_for_game_over()
        
    def getReward(self):
        return self.reward
    
    # def render(self):
        # return self.game.render()