import numpy as np
import math
from .src.game2048 import Game2048
from .Game import Game
import pygame


class Puzzle2048(Game):
    def __init__(self, size: int = 4):
        super().__init__()
        self.name: str = "game2048"
        self.size: int = size
        self.game: Game2048 = Game2048(self.size)
        self.observationShape: tuple = (1, self.size, self.size)
        self.actionSpace: int = 4
        self.reward: float = 0
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

    def reset(self):
        self.game = Game2048(self.size)
        return self.getState()

    def getState(self):
        state = np.zeros(self.observationShape, dtype=int)
        for _, _, cell in self.game.grid.eachCell():
            if cell:
                state[0][cell.x][cell.y] = math.log2(cell.value)
        return state

    def getActionMask(self, state):
        actions = [
            (0, -1),  # Up
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0)  # Left
        ]
        mask = np.zeros(self.actionSpace, dtype=int)
        for i, vector in enumerate(actions):
            for x, col in enumerate(state):
                for y, cell in enumerate(col):
                    if not cell == 0:
                        if x + vector[0] >= 0 and x + vector[0] < self.size and \
                            y + vector[1] >= 0 and y + vector[1] < self.size:
                            next = state[x + vector[0]][y + vector[1]]
                            if next == 0 or next == cell:
                                mask[i] = 1
                                break
        return mask

    def takeAction(self, action):
        score = self.game.score
        moved = self.game.move(action)
        if not moved:
            raise Exception()
        self.reward = self.game.score - score
        # self.reward = max(-1, min(1, self.reward))
        return super().takeAction(action)

    def getDone(self) -> bool:
        return self.game.isGameTerminated()

    def getReward(self) -> float:
        return self.reward

    def render(self) -> None:
        if self.rendered:
            return

        self.rendered = True
        self.width = 500
        self.height = 600
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.name)
        pygame.init()

    def update(self) -> None:
        if not self.rendered:
            return

        pygame.event.get()
        state = self.getState()
        font = pygame.font.Font(pygame.font.get_default_font(), 36)
        scoreHeight = 100

        score_rect = pygame.Rect(0, 0, self.width, scoreHeight)
        pygame.draw.rect(self.display, (0, 0, 0), score_rect)
        text = font.render(str(self.game.score), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = score_rect.center
        self.display.blit(text, text_rect)
        for x, col in enumerate(state[0]):
            for y, cell in enumerate(col):
                block_size = ((self.height - scoreHeight) / self.observationShape[1], self.width / self.observationShape[2])
                block_rect = pygame.Rect(x * block_size[1], scoreHeight + y * block_size[0], block_size[1], block_size[0])
                pygame.draw.rect(self.display, self.tileColors[2 ** cell], block_rect)
                if not cell == 0:
                    text = font.render(str(2 ** cell), True, (0, 0, 0))
                    text_rect = text.get_rect()
                    text_rect.center = block_rect.center
                    self.display.blit(text, text_rect)
        pygame.display.update()
