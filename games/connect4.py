import numpy as np

from .tictactoe import TicTacToe


class Connect4(TicTacToe):
    def __init__(self, size_x=7, size_y=6):
        super().__init__(size_x, size_y, 4)
        self.name: str = "Connect4"

    @property
    def action_spaces(self):
        return range(self.sizeX)

    def _step(self, player_id, action):
        posX = action
        posY = -1
        for y, cell in enumerate(self.game.cells[posX]):
            if cell == 0:
                posY = y
        if posY == -1:
            raise Exception("Invalid Move")
        
        action = posX * self.sizeY + posY
        self.game.step(player_id, action)

    def get_mask(self, player_id):
        mask = np.zeros(self.action_spaces.length, dtype=int)
        for x, rows in enumerate(self.game.cells):
            if rows[0] == 0:
                mask[x] = 1
        return mask
