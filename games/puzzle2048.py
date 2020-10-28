import numpy as np
import math
from .src.game2048 import Game2048 as Src
from .Game import Game
import pygame


class Puzzle2048(Game):
    def __init__(self, size: int = 4):
        super().__init__()
        self.name: str = "game2048"
        self.size: int = size
        self.seed = None
        self.game = Src(self.size, self.seed)
        self.observationShape: tuple = (1, self.size, self.size)

    @property
    def players(self):
        return [1]
      
    @property
    def actionSpaces(self):
        return [
            (0, -1),  # Up
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0)  # Left
        ]

    def canStep(self, playerId):
        return True

    def reset(self):
        self.game = Src(self.size, self.seed)

    def getState(self, playerId: int):
        state = np.zeros(self.observationShape, dtype=float)
        # state = np.zeros((1, self.size, self.size), dtype=int)
        for _, _, cell in self.game.grid.eachCell():
            if cell:
                state[0][cell.x][cell.y] = math.log2(cell.value)
        return state

    def getMask(self, playerId: int):
        state = self.game.grid.cells
        mask = np.zeros(self.actionCount, dtype=bool)
        for i, vector in enumerate(self.actionSpaces):
            for x, col in enumerate(state):
                if mask[i]:
                    break
                for y, cell in enumerate(col):
                    if cell:
                        if x + vector[0] >= 0 and x + vector[0] < self.size and \
                            y + vector[1] >= 0 and y + vector[1] < self.size:
                            next = state[x + vector[0]][y + vector[1]]
                            if next is None or next.value == cell.value:
                                mask[i] = True
                                break
        return mask

    def _step(self, playerId: int, action) -> None:
        score = self.game.score
        moved = self.game.move(action)
        if not moved:
            raise Exception("Invalid move")
        self.reward[playerId] = self.game.score - score

    def isDone(self) -> bool:
        return self.game.isGameTerminated()

    def render(self) -> None:
        self.renderer = Renderer(self)

    def update(self) -> None:
        if self.renderer is None:
            return
        self.renderer.update()


class Renderer:
    def __init__(self, game):
        self.game = game
        self.width = 500
        self.height = 600
        self.scoreHeight = 100
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.game.name)
        pygame.init()
        self.tileColors: dict = {
            1: (204, 192, 179),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (244, 149, 99),
            32: (245, 121, 77),
            64: (245, 93, 55),
            128: (238, 232, 99),
            256: (237, 176, 77),
            512: (236, 176, 77),
            1024: (235, 148, 55),
            2048: (234, 120, 33),
            4096: (234, 120, 33),
            8192: (234, 120, 33),
            16384: (234, 120, 33),
            32768: (234, 120, 33),
            65536: (234, 120, 33),
        }

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
        for x, col in enumerate(self.game.game.grid.cells):
            for y, cell in enumerate(col):
                block_size = ((self.height - self.scoreHeight) / self.game.game.size, self.width / self.game.game.size)
                block_rect = pygame.Rect(x * block_size[1], self.scoreHeight + y * block_size[0], block_size[1], block_size[0])
                if cell is not None:
                    pygame.draw.rect(self.display, self.tileColors[cell.value], block_rect)
                    text = font.render(str(cell.value), True, (0, 0, 0))
                    text_rect = text.get_rect()
                    text_rect.center = block_rect.center
                    self.display.blit(text, text_rect)
        pygame.display.update()
