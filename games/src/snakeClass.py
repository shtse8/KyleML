import math
import numpy as np
from enum import Enum


class Direction(Enum):
    Up = 0
    Left = 1
    Down = 2
    Right = 3


class Action(Enum):
    Right = -1
    Left = 1
    Void = 0


class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class GameObject:
    def __init__(self):
        self.coord = None

    def setCoord(self, coord):
        self.coord = coord


class Block(GameObject):
    def __init__(self):
        super().__init__()


class Player(GameObject):
    def __init__(self, direction=Direction.Right):
        super().__init__()
        self.direction = direction
        self.length = 0
        self.nextBody: PlayerBody = None


class PlayerBody(GameObject):
    def __init__(self):
        super().__init__()
        self.nextBody: PlayerBody = None


class Food(GameObject):
    def __init__(self):
        super().__init__()


class Grid:
    def __init__(self, width, height, rng):
        self.width = width
        self.height = height
        self.rng = rng
        self.grid = np.empty((width, height), dtype=object)

    def getRandomEmpty(self):
        emptySpaces = []
        for x, col in enumerate(self.grid):
            for y, cell in enumerate(col):
                if cell is None:
                    emptySpaces.append(Coord(x, y))
        return self.rng.choice(emptySpaces)

    def set(self, obj: GameObject, coord: Coord=None):
        if coord is None:
            coord = self.getRandomEmpty()

        # if self.get(coord) is not None:
        #     raise Exception("Block is not empty.")

        self.grid[coord.x, coord.y] = obj
        obj.setCoord(coord)

    def get(self, coord: Coord):
        return self.grid[coord.x, coord.y]

    def clear(self, coord: Coord):
        self.grid[coord.x, coord.y] = None

    def isValidCoord(self, coord: Coord):
        return coord.x >= 0 and coord.x < self.grid.width and coord.y >= 0 and coord.y < self.grid.height

class Game:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dangerObjects = (Block, PlayerBody)

        # reset
        self.score = 0
        self.isEnd = False
        self.grid = Grid(self.width, self.height, self.rng)

        # Walls
        for x in range(self.width):
            self.grid.set(Block(), Coord(x, 0))
            self.grid.set(Block(), Coord(x, self.height - 1))

        for y in range(1, self.height - 1):
            self.grid.set(Block(), Coord(0, y))
            self.grid.set(Block(), Coord(self.width - 1, y))

        # Player
        self.player = Player()
        self.grid.set(self.player, Coord(
            x=math.floor((self.width - 1) * 0.45),
            y=math.floor((self.height - 1) * 0.5)))

        # Food
        self.food = Food()
        self.grid.set(self.food)

    def isDanger(self, action: Action):
        return isinstance(self.grid.get(self._getTargetCoord(self._getNewDirection(action))), self.dangerObjects)

    def _getNewDirection(self, action: Action):
        return Direction((self.player.direction.value + action.value) % len(Direction))

    def _getTargetCoord(self, direction: Direction = None):
        if direction is None:
            direction = self.player.direction

        # Move Player Head
        if direction == Direction.Up:
            targetCoord = Coord(self.player.coord.x, self.player.coord.y - 1)
        elif direction == Direction.Down:
            targetCoord = Coord(self.player.coord.x, self.player.coord.y + 1)
        elif direction == Direction.Left:
            targetCoord = Coord(self.player.coord.x - 1, self.player.coord.y)
        elif direction == Direction.Right:
            targetCoord = Coord(self.player.coord.x + 1, self.player.coord.y)

        return targetCoord

    def step(self, action: Action):
        if self.isEnd:
            raise Exception("Game Over.")
        
        self.player.direction = self._getNewDirection(action)
        self.grid.clear(self.player.coord)

        # Move Player Body
        coord = self.player.coord
        curLen = 0
        playerBody = self.player
        while playerBody.nextBody is not None:
            playerBody = playerBody.nextBody
            coord = self._moveTo(playerBody, coord)
            curLen += 1
        
        # move player Head and detect collision
        targetCoord = self._getTargetCoord()
        targetObj = self.grid.get(targetCoord)
        if targetObj is not None:
            if isinstance(targetObj, self.dangerObjects):
                self.isEnd = True
            elif isinstance(targetObj, Food):
                self.score += 1
                self.player.length += 1
                self.food = Food()
                self.grid.set(self.food)
        self.grid.set(self.player, targetCoord)

        # every step add one body item
        if curLen < self.player.length:
            playerBody.nextBody = PlayerBody()
            self.grid.set(playerBody.nextBody, coord)
            
    def _moveTo(self, obj: GameObject, coord: Coord):
        oldCoord = obj.coord
        self.grid.set(obj, coord)
        self.grid.clear(oldCoord)
        return oldCoord
