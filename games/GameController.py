from .Game import Game


class GameController(object):
    def __init__(self, game: Game):
        self.game: Game = game

    def newGame(self):
        return self.game()
