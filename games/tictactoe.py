import math
import numpy as np
import pygame

from .Game import Game
from .src.tictactoe import TicTacToe as Src


class TicTacToe(Game):
    def __init__(self, size_x: int = 3, size_y: int = 3, win_size: int = 3):
        super().__init__()
        self.name: str = "tictactoe"
        self.sizeX = size_x
        self.sizeY = size_y
        self.winSize = win_size
        self.game = Src(size_x, size_y, win_size)
        self.observationShape: tuple = (self.player_count, size_x, size_y)

    @property
    def players(self):
        return [1, 2]
      
    @property
    def action_spaces(self):
        return range(self.sizeX * self.sizeY)

    def get_player_count(self):
        return self.player_count

    def reset(self):
        self.game = Src(self.sizeX, self.sizeY, self.winSize)

    def get_done_reward(self, player_id) -> float:
        if not self.is_done():
            raise Exception("Game is not finished.")
        if self.game.winner == 0:
            return 0
        elif self.game.winner == player_id:
            return 1
        else:
            return -1

    def can_step(self, player_id):
        return not self.is_done() and player_id == self.game.turn

    def get_state(self, player_id):
        # 0 - Empty
        # 1 - Self
        # 2 - Opponent
        state = np.zeros(self.observationShape, dtype=int)
        for x, rows in enumerate(self.game.cells):
            for y, cell in enumerate(rows):
                if cell == player_id:
                    state[0][x][y] = 1
                elif cell != 0:
                    state[1][x][y] = 1
        return state

    def get_mask(self, player_id):
        mask = np.zeros(self.action_spaces, dtype=bool)
        for i in range(self.action_spaces):
            vector = self.game.toVector(i)
            cell = self.game.getCell(vector)
            if cell == 0:
                mask[i] = True
        return mask

    def _step(self, player_id, action):
        self.game.step(player_id, action)

    def is_done(self) -> bool:
        return self.game.isEnd
