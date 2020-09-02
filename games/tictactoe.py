import numpy as np
import math
from .src.tictactoe import TicTacToe as Src
from .Game import Game
import pygame


class Puzzle2048(Game):
    def __init__(self, size: int = 4):
        super().__init__()
        self.name: str = "tictactoe"
        self.game = Src()
        self.size = self.game.size
        self.observationShape: tuple = (1, self.size, self.size)
        self.actionSpace: int = self.size * self.size
        self.reward: float = 0

    def reset(self):
        self.game = Src(self.size)

    def getState(self):
        return self.game.cells

    def getActionMask(self, state):
        mask = np.zeros(self.actionSpace, dtype=int)
        for a in self.actionSpace:
            vector = self.game.toVector(a)
            cell = self.game.getCell(vector)
            if cell == 0:
                mask = 1
        return mask

    def takeAction(self, action):
        result = self.game.step(action)
        if result:
            # reward for both players?
            self.reward = 1

    def getDone(self) -> bool:
        return self.game.isEnd

    def getReward(self) -> float:
        return self.reward
