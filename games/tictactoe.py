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

    def getState(self, playerId):
        # 0 - Empty
        # 1 - Self
        # 2 - Opponent
        state = np.zeros(self.observationShape)
        for x, rows in enumerate(self.game.cells):
            for y, cell in enumerate(rows):
                if cell == playerId:
                    state[0][x][y] = 1
                elif cell != 0:
                    state[0][x][y] = 2
        return state

    def getMask(self, playerId, state):
        mask = np.zeros(self.actionSpace, dtype=int)
        for i in range(len(self.actionSpace)):
            vector = self.game.toVector(i)
            cell = self.game.getCell(vector)
            if cell == 0:
                mask[i] = 1
        return mask

    def step(self, playerId, action):
        result = self.game.step(playerId, action)
        if result:
            # reward for both players?
            self.reward = 1

    def getDone(self) -> bool:
        return self.game.isEnd

    def getReward(self) -> float:
        return self.reward
