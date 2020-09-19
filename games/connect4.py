from .tictactoe import TicTacToe
import numpy as np

class Connect4(TicTacToe):
    def __init__(self, sizeX=7, sizeY=6):
        super().__init__(sizeX, sizeY, 4)
        self.name: str = "Connect4"
        self.actionSpace: int = sizeX

    def _step(self, playerId, action):
        posX = action
        posY = -1
        for y, cell in enumerate(self.game.cells[posX]):
            if cell == 0:
                posY = y
        if posY == -1:
            raise Exception("Invalid Move")
        
        action = posX * self.sizeY + posY
        self.game.step(playerId, action)

    def getMask(self, playerId):
        mask = np.zeros(self.actionSpace, dtype=int)
        for x, rows in enumerate(self.game.cells):
            if rows[0] == 0:
                mask[x] = 1
        return mask
