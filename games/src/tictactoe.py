from __future__ import annotations
import numpy as np


class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __iadd__(self, other: Vector):
        return self.__add__(other)

    def __isub__(self, other: Vector):
        return self.__sub__(other)

    def __add__(self, other: Vector):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, multiplier: float):
        return Vector(self.x * multiplier, self.y * multiplier)

    def __div__(self, multiplier: float):
        return Vector(self.x / multiplier, self.y / multiplier)


class TicTacToe:
    def __init__(self, sizeX=3, sizeY=3, winSize=3, players=2) -> None:
        self.players = players
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.winSize = winSize
        self.cells = np.zeros((self.sizeX, self.sizeY))
        self.turn = 1
        self.winner = 0
        self.isEnd = False

    def step(self, playerId, pos: int) -> None:
        if self.isEnd:
            raise Exception("Game over.")

        if self.turn != playerId:
            raise Exception("Not your turn.")

        vector = self.toVector(pos)
        if self.cells[vector.x][vector.y] != 0:
            raise Exception("Invalid Move")

        self.setCell(vector, playerId)
        if self.checkWin(vector):
            self.winner = self.turn
            self.isEnd = True
        elif 0 not in self.cells.flatten():
            self.isEnd = True
        else:
            # next turn
            self.turn = self.turn % self.players + 1

    def setCell(self, vector: Vector, id: int) -> None:
        self.cells[vector.x][vector.y] = id

    def getCell(self, vector: Vector):
        return self.cells[vector.x][vector.y]

    def isInbound(self, vector: Vector):
        return vector.x >= 0 and vector.x < self.sizeX and vector.y >= 0 and vector.y < self.sizeY

    def toVector(self, pos: int) -> Vector:
        x = pos // self.sizeY
        y = pos % self.sizeY
        return Vector(x, y)

    def checkWin(self, original: Vector):
        mark = self.getCell(original)
        if mark == 0:
            raise Exception("Original is empty")

        transversals = [
            Vector(-1, -1),  # \
            Vector(-1, 0),  # -
            Vector(-1, 1),  # /
            Vector(0, -1)  # |
        ]
        for transversal in transversals:
            len = 1
            # forward and backward checkings
            for direction in [1, -1]:
                v = original
                t = transversal * direction
                while True:
                    v += t
                    if not self.isInbound(v) or self.getCell(v) != mark:
                        break
                    len += 1
                    if len >= self.winSize:
                        return True

        return False
