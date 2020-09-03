import numpy as np
import math
from .src.tictactoe import TicTacToe as Src
from .Game import Game
import pygame


class TicTacToe(Game):
    def __init__(self, size: int = 4):
        super().__init__()
        self.name: str = "tictactoe"
        self.game = Src()
        self.size = self.game.size
        self.observationShape: tuple = (1, self.size, self.size)
        self.actionSpace: int = self.size * self.size
        self.reward: float = 0

    def getPlayerCount(self):
        return 2

    def reset(self):
        self.game = Src()

    def getDoneReward(self, playerId) -> float:
        if not self.isDone():
            raise Exception("Game is not finished.")
        if self.game.winner == 0:
            return 0
        elif self.game.winner == playerId:
            return 1
        else:
            return -1

    def canStep(self, playerId):
        return not self.isDone() and playerId == self.game.turn

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
        for i in range(self.actionSpace):
            vector = self.game.toVector(i)
            cell = self.game.getCell(vector)
            if cell == 0:
                mask[i] = 1
        return mask

    def _step(self, playerId, action):
        self.game.step(playerId, action)

    def isDone(self) -> bool:
        return self.game.isEnd

    def getReward(self) -> float:
        return self.reward
