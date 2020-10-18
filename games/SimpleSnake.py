import numpy as np
from .src.snakeClass import Game as GameSrc, Action, Direction, PlayerBody, Block, Player, Food
from .Game import Game
import pygame

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
        return np.array([
            self.game.isEnd or self.game.isDanger(Action.Left),  # danger Left
            self.game.isEnd or self.game.isDanger(Action.Right),  # danger Right
            self.game.isEnd or self.game.isDanger(Action.Void),  # danger Straight
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

    def render(self) -> None:
        self.renderer = SnakeRenderer(self)

    def update(self) -> None:
        if self.renderer is None:
            return
        self.renderer.update()


class SnakeRenderer:
    def __init__(self, game):
        self.game = game
        self.width = 500
        self.height = 600
        self.scoreHeight = 100
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.game.name)
        pygame.init()

    def update(self):
        pygame.event.get()
        bg_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.display, (0, 0, 0), bg_rect)

        # state = self.getState()
        font = pygame.font.Font(pygame.font.get_default_font(), 36)
        score_rect = pygame.Rect(0, 0, self.width, self.scoreHeight)
        text = font.render(str(self.game.game.score), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = score_rect.center
        self.display.blit(text, text_rect)
        
        # grid = np.zeros((22, 22), dtype=int)
        for x, col in enumerate(self.game.game.grid.grid):
            for y, cell in enumerate(col):
                block_size = ((self.height - self.scoreHeight) // self.game.game.height, self.width // self.game.game.width)
                block_rect = pygame.Rect(x * block_size[1], self.scoreHeight + y * block_size[0], block_size[1], block_size[0])
                if isinstance(cell, Block):
                    pygame.draw.rect(self.display, (255, 255, 255), block_rect)
                elif isinstance(cell, Food):
                    pygame.draw.rect(self.display, (0, 255, 0), block_rect)
                elif isinstance(cell, (Player, PlayerBody)):
                    pygame.draw.rect(self.display, (0, 0, 255), block_rect)
        pygame.display.update()
    
