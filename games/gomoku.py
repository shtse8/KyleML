from .tictactoe import TicTacToe


class Gomoku(TicTacToe):
    def __init__(self):
        super().__init__(15, 5)
        self.name: str = "Gomoku"
