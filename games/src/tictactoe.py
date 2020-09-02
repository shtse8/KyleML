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
    def __init__(self) -> None:
        self.size = 3
        self.cells = np.zeros((self.size, self.size))
        self.turn = 0

    def step(self, playerId, pos: int) -> None:
        if self.turn != playerId:
            raise Exception("Not your turn.")

        vector = self.toVector(pos)
        self.cells[vector.x][vector.y]
        if self.cells[vector.x][vector.y] != 0:
            raise Exception("Invalid Move")

        self.setCell(vector, playerId)
        if self.checkWin(vector):
            return True

        return False

    def setCell(self, vector: Vector, id: int) -> None:
        self.cells[vector.x][vector.y] = id

    def getCell(self, vector: Vector):
        return self.cells[vector.x][vector.y]

    def isInbound(self, vector: Vector):
        return vector.x >= 0 and vector.x < self.size and vector.y >= 0 and vector.y < self.size

    def toVector(self, pos: int) -> Vector:
        x = pos // self.size
        y = pos % self.size
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
                    if len >= self.size:
                        return True

        return False
