import numpy as np

from operator import add
from .src.snakeClass import Game as GameSrc
from .Game import Game

class SimpleSnake(Game):
    def __init__(self):
        super().__init__()
        self.name = "SimpleSnake"
        self.game = GameSrc(22, 22, seed=0)
        self.observationShape = 11
        self.actionSpace = 3
    
    def getPlayerCount(self):
        return 1

    def canStep(self, playerId):
        return True

    def reset(self):
        self.game = GameSrc(22, 22, seed=0)
        self.game.food.x = self.game.player.x + 2
        self.game.food.y = self.game.player.y
        
    def getState(self, playerId):
        state = np.array([
            (self.game.player.x_change == 1 and self.game.player.y_change == 0 and ((list(map(add, self.game.player.position[-1], [1, 0])) in self.game.player.position) or
            self.game.player.position[-1][0] + 1 >= (self.game.width - 1))) or (self.game.player.x_change == -1 and self.game.player.y_change == 0 and ((list(map(add, self.game.player.position[-1], [-1, 0])) in self.game.player.position) or
            self.game.player.position[-1][0] - 1 < 1)) or (self.game.player.x_change == 0 and self.game.player.y_change == -1 and ((list(map(add, self.game.player.position[-1], [0, -1])) in self.game.player.position) or
            self.game.player.position[-1][-1] - 1 < 1)) or (self.game.player.x_change == 0 and self.game.player.y_change == 1 and ((list(map(add, self.game.player.position[-1], [0, 1])) in self.game.player.position) or
            self.game.player.position[-1][-1] + 1 >= (self.game.height-1))),  # danger straight

            (self.game.player.x_change == 0 and self.game.player.y_change == -1 and ((list(map(add,self.game.player.position[-1],[1, 0])) in self.game.player.position) or
            self.game.player.position[ -1][0] + 1 > (self.game.width-1))) or (self.game.player.x_change == 0 and self.game.player.y_change == 1 and ((list(map(add,self.game.player.position[-1],
            [-1,0])) in self.game.player.position) or self.game.player.position[-1][0] - 1 < 1)) or (self.game.player.x_change == -1 and self.game.player.y_change == 0 and ((list(map(
            add,self.game.player.position[-1],[0,-1])) in self.game.player.position) or self.game.player.position[-1][-1] - 1 < 1)) or (self.game.player.x_change == 1 and self.game.player.y_change == 0 and (
            (list(map(add,self.game.player.position[-1],[0,1])) in self.game.player.position) or self.game.player.position[-1][
             -1] + 1 >= (self.game.height-1))),  # danger right

             (self.game.player.x_change == 0 and self.game.player.y_change == 1 and ((list(map(add,self.game.player.position[-1],[1,0])) in self.game.player.position) or
             self.game.player.position[-1][0] + 1 > (self.game.width-1))) or (self.game.player.x_change == 0 and self.game.player.y_change == -1 and ((list(map(
             add, self.game.player.position[-1],[-1,0])) in self.game.player.position) or self.game.player.position[-1][0] - 1 < 1)) or (self.game.player.x_change == 1 and self.game.player.y_change == 0 and (
            (list(map(add,self.game.player.position[-1],[0,-1])) in self.game.player.position) or self.game.player.position[-1][-1] - 1 < 1)) or (
            self.game.player.x_change == -1 and self.game.player.y_change == 0 and ((list(map(add,self.game.player.position[-1],[0,1])) in self.game.player.position) or
            self.game.player.position[-1][-1] + 1 >= (self.game.height-1))), #danger left


            self.game.player.x_change == -1,  # move left
            self.game.player.x_change == 1,  # move right
            self.game.player.y_change == -1,  # move up
            self.game.player.y_change == 1,  # move down
            self.game.food.x < self.game.player.x,  # self.game.food left
            self.game.food.x > self.game.player.x,  # self.game.food right
            self.game.food.y < self.game.player.y,  # self.game.food up
            self.game.food.y > self.game.player.y  # self.game.food down
        
        ]).astype(int)
        
        # state = np.array([
            # self.game.player.x,
            # self.game.player.y,
            # self.game.player.x_change,
            # self.game.player.y_change,
            # self.game.food.x,
            # self.game.food.y,
            # self.game.width,
            # self.game.height
        # ]).astype(int)
        # state = np.concatenate((state, np.array(self.game.player.position).flatten()), axis = 0)
        # state = np.pad(state, (0, self.observationShape - len(state)))
        return state
        
    def getMask(self, playerId, state):
        return np.ones(self.actionSpace)

    def _step(self, playerId, action):
        action_array = np.zeros(self.actionSpace)
        action_array[action] = 1
        self.game.player.do_move(action_array, self.game)
        self.game.display()
        self.reward[playerId] = 1 if self.game.player.eaten else 0
        
    def isDone(self):
        return self.game.end

    
    