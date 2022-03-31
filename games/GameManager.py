from typing import Iterator

from utils.Event import Event
from .SimpleSnake import SimpleSnake
from .Snake import Snake
from .connect4 import Connect4
from .gomoku import Gomoku
# from .mario import Mario
from .gymgame import GymGame
from .puzzle2048 import Puzzle2048
from .tictactoe import TicTacToe
from .Game import Game, Renderer, GetOnlySimpleQueue, GameEvent


class GameContainer:
    def __init__(self, name: str):
        self.__name = name
        self.__game: Game = None
        self.__renderer: Renderer = None
        self.__on_done = Event()
        self.reset()

    @property
    def on_done(self) -> Event:
        return self.__on_done

    @property
    def name(self) -> str:
        return self.__name

    def __set_game(self, game: Game):
        if game is None:
            raise ValueError
        self.__game = game
        if self.__renderer is not None:
            self.__renderer.game = self.__game

    def reset(self):
        if self.__name == "snake":
            self.__set_game(Snake())
        elif self.__name == "simple-snake":
            self.__set_game(SimpleSnake())
        elif self.__name == "cartpole":
            self.__set_game(GymGame("CartPole-v0"))
        elif self.__name == "2048":
            self.__set_game(Puzzle2048())
        # elif self.name == "mario":
        #     self._game = Mario()
        elif self.__name == "pong":
            self.__set_game(GymGame("Pong-v0"))
        elif self.__name == "breakout":
            self.__set_game(GymGame("Breakout-v0"))
        elif self.__name == "blackjack":
            self.__set_game(GymGame("Blackjack-v0"))
        elif self.__name == "pacman":
            self.__set_game(GymGame("MsPacman-v0"))
        elif self.__name == "simple-pacman":
            self.__set_game(GymGame("MsPacman-ram-v0"))
        elif self.__name == "pinball":
            self.__set_game(GymGame("VideoPinball-v0"))
        elif self.__name == "simple-pinball":
            self.__set_game(GymGame("VideoPinball-ram-v0"))
        elif self.__name == "tictactoe":
            self.__set_game(TicTacToe())
        elif self.__name == "gomoku":
            self.__set_game(Gomoku())
        elif self.__name == "connect4":
            self.__set_game(Connect4())
        else:
            raise ValueError("Unknown Game " + self.__name)

    @property
    def on_done(self) -> Event:
        return self._on_done

    @property
    def observation_shape(self) -> tuple:
        return self.__game.observation_shape

    @property
    def action_spaces(self):
        return self.__game.action_spaces

    @property
    def action_count(self) -> int:
        return self.__game.action_count

    @property
    def player_count(self) -> int:
        return self.__game.player_count

    @property
    def is_done(self) -> bool:
        return self.__game.is_done

    @property
    def players(self):
        return self.__game.players

    def get_events(self) -> Iterator[GameEvent]:
        for event in self.__renderer.event.get():
            yield self.__game.process_event(event)

    def render(self):
        if self.__renderer is not None:
            return

        if self.__name == "2048":
            self.__renderer = Renderer()
            self.__renderer.game = self.__game
        else:
            raise ValueError("Unknown Game " + self.__name)

    def update(self):
        self.__renderer.update()

    def Run(self):
        while not self.__game.is_done:
            self.update()


class GameManager:
    def __init__(self, name: str):
        self.name: str = name

    def create(self) -> GameContainer:
        return GameContainer(self.name)
