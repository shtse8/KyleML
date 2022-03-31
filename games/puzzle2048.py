import math
import math
import queue
from collections.abc import Sequence
from typing import Any

import numpy as np
import pygame

from .Game import Game, Renderer, Player, P, RendererEvent, GameEvent, GameEventType
from .src.game2048 import Game2048 as GameCore


class Puzzle2048Player(Player):

    @property
    def score(self) -> int:
        return self.game.core.score

    @property
    def can_step(self) -> bool:
        return self.is_done

    @property
    def state(self):
        state = np.zeros(self.game.observation_shape, dtype=float)
        # state = np.zeros((1, self.size, self.size), dtype=int)
        for _, _, cell in self.game.core.grid.each_cell():
            if cell:
                state[0][cell.x][cell.y] = math.log2(cell.value)
                # state[0][cell.x][cell.y] = 1
                # state[int(math.log2(cell.value))][cell.x][cell.y] = 1
        return state

    @property
    def action_spaces_mask(self):
        state = self.game.core.grid.cells
        mask = np.zeros(self.game.action_count, dtype=bool)
        for i, vector in enumerate(self.game.action_spaces):
            for x, col in enumerate(state):
                if mask[i]:
                    break
                for y, cell in enumerate(col):
                    if cell:
                        if 0 <= x + vector[0] < self.game.size and \
                                0 <= y + vector[1] < self.game.size:
                            next_cell = state[x + vector[0]][y + vector[1]]
                            if next_cell is None or next_cell.value == cell.value:
                                mask[i] = True
                                break
        return mask

    @property
    def is_done(self) -> bool:
        return self.game.core.is_game_terminated()

    def step(self, action) -> float:
        last_score: int = self.score
        self.game.core.move(action)
        return self.score - last_score


class Puzzle2048(Game[Puzzle2048Player]):

    def __init__(self, size: int = 4):
        super().__init__("game2048")
        self.size: int = size
        self.seed: str = None
        self.core: GameCore = GameCore(self.size, self.seed)

    @property
    def player_count(self) -> int:
        return 1

    @property
    def action_spaces(self) -> Sequence[Any]:
        return [
            (0, -1),  # Up
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0)  # Left
        ]

    @property
    def observation_shape(self) -> tuple:
        # return (12, self.size, self.size)
        return 1, self.size, self.size

    @property
    def is_done(self) -> bool:
        return self.core.is_game_terminated()

    def _create_player(self, player_id: int) -> P:
        return Puzzle2048Player(self, player_id)

    def update_display(self, surface: pygame.Surface):
        tile_colors: dict = {
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
        width: int = surface.get_width()
        height: int = surface.get_height()
        score_height: int = 100

        # Draw UI
        bg_rect = pygame.Rect(0, 0, width, height)
        pygame.draw.rect(surface, (0, 0, 0), bg_rect)

        # state = self.getState()
        font = pygame.font.Font(pygame.font.get_default_font(), 36)
        score_rect = pygame.Rect(0, 0, width, score_height)
        text = font.render(str(self.core.score), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = score_rect.center

        surface.blit(text, text_rect)
        for x, col in enumerate(self.core.grid.cells):
            for y, cell in enumerate(col):
                block_size = (
                    (height - score_height) / self.core.size, width / self.core.size)
                block_rect = pygame.Rect(x * block_size[1], score_height + y * block_size[0], block_size[1],
                                         block_size[0])
                if cell is not None:
                    pygame.draw.rect(surface, tile_colors[cell.value], block_rect)
                    text = font.render(str(cell.value), True, (0, 0, 0))
                    text_rect = text.get_rect()
                    text_rect.center = block_rect.center
                    surface.blit(text, text_rect)

    def process_event(self, event: RendererEvent) -> GameEvent:
        if event == RendererEvent.UP:
            return GameEvent(GameEventType.Step, 0, 0)
        elif event == RendererEvent.RIGHT:
            return GameEvent(GameEventType.Step, 0, 1)
        elif event == RendererEvent.DOWN:
            return GameEvent(GameEventType.Step, 0, 2)
        elif event == RendererEvent.LEFT:
            return GameEvent(GameEventType.Step, 0, 3)
