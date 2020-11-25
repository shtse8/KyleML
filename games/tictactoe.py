import numpy as np
import math
from .src.tictactoe import TicTacToe as Src
from .Game import Game
import pygame


class TicTacToe(Game):
    def __init__(self, sizeX: int = 3, sizeY: int = 3, winSize: int = 3):
        super().__init__()
        self.name: str = "tictactoe"
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.winSize = winSize
        self.game = Src(sizeX, sizeY, winSize)
        self.observationShape: tuple = (self.playerCount, sizeX, sizeY)

    @property
    def players(self):
        return [1, 2]
      
    @property
    def actionSpaces(self):
        return range(self.sizeX * self.sizeY)

    def getPlayerCount(self):
        return self.playerCount

    def reset(self):
        self.game = Src(self.sizeX, self.sizeY, self.winSize)

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
        state = np.zeros(self.observationShape, dtype=int)
        for x, rows in enumerate(self.game.cells):
            for y, cell in enumerate(rows):
                if cell == playerId:
                    state[0][x][y] = 1
                elif cell != 0:
                    state[1][x][y] = 1
        return state

    def getMask(self, playerId):
        mask = np.zeros(self.actionSpace, dtype=bool)
        for i in range(self.actionSpace):
            vector = self.game.toVector(i)
            cell = self.game.getCell(vector)
            if cell == 0:
                mask[i] = True
        return mask

    def _step(self, playerId, action):
        self.game.step(playerId, action)

    def isDone(self) -> bool:
        return self.game.isEnd
