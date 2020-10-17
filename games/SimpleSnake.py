import numpy as np
from .src.snakeClass import Game as GameSrc, Action, Direction, PlayerBody, Block, Player, Food
from .Game import Game


class SimpleSnake(Game):
    def __init__(self):
        super().__init__()
        self.name = "SimpleSnake"
        self.seed = None
        self.game = GameSrc(22, 22, self.seed)
        self.observationShape = 11

    @property
    def players(self):
        return [0]

    @property
    def actionSpaces(self):
        return [x for x in Action]

    def canStep(self, playerId):
        return True

    def reset(self):
        self.game = GameSrc(22, 22, self.seed)
        
    def getState(self, playerId):
        # grid = np.zeros((22, 22), dtype=int)
        # for x, col in enumerate(self.game.grid.grid):
        #     for y, cell in enumerate(col):
        #         if isinstance(cell, Block):
        #             grid[x, y] = '9'
        #         elif isinstance(cell, Food):
        #             grid[x, y] = '3'
        #         elif isinstance(cell, Player):
        #             grid[x, y] = '1'
        #         elif isinstance(cell, PlayerBody):
        #             grid[x, y] = '2'
        # print(grid)

        return np.array([
            self.game.isEnd or isinstance(self.game.grid.get(self.game.getTargetCoord(Action.Left)), self.game.dangerObjects),  # danger Left
            self.game.isEnd or isinstance(self.game.grid.get(self.game.getTargetCoord(Action.Right)), self.game.dangerObjects),  # danger Right
            self.game.isEnd or isinstance(self.game.grid.get(self.game.getTargetCoord(Action.Void)), self.game.dangerObjects),  # danger Straight
            self.game.player.direction == Direction.Left,  # move left
            self.game.player.direction == Direction.Right,  # move right
            self.game.player.direction == Direction.Up,  # move up
            self.game.player.direction == Direction.Down,  # move down
            self.game.food.coord.x < self.game.player.coord.x,  # self.game.food left
            self.game.food.coord.x > self.game.player.coord.x,  # self.game.food right
            self.game.food.coord.y < self.game.player.coord.y,  # self.game.food up
            self.game.food.coord.y > self.game.player.coord.y  # self.game.food down
        ]).astype(int)
        
    def getMask(self, playerId: int):
        return np.ones(self.actionCount)

    def _step(self, playerId, action):
        score = self.game.score
        self.game.step(self.actionSpaces[action])
        self.reward[playerId] = self.game.score - score
        
    def isDone(self):
        return self.game.isEnd

    
    