from enum import Enum
from queue import SimpleQueue
from abc import abstractmethod, ABCMeta
from typing import TypeVar, Generic, Any

import pygame
from pygame.event import Event


class Player(metaclass=ABCMeta):
    def __init__(self, game, player_id: int):
        self.game = game
        self.player_id = player_id

    @property
    @abstractmethod
    def can_step(self) -> bool:
        pass

    @property
    @abstractmethod
    def action_spaces_mask(self):
        pass

    @property
    @abstractmethod
    def score(self) -> float:
        pass

    @property
    @abstractmethod
    def is_done(self) -> bool:
        pass

    # Return reward
    @abstractmethod
    def step(self, action) -> float:
        pass


class GameEventType(Enum):
    Step = 0


class GameEvent:
    def __init__(self, event_type: GameEventType, player_id: int, value: Any):
        self.event_type = event_type
        self.player_id = player_id
        self.value = value


P = TypeVar('P', bound=Player)


class Game(Generic[P], metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name
        self._players: list[P] = None

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_spaces(self):
        pass

    @property
    def action_count(self) -> int:
        return len(self.action_spaces)

    @property
    @abstractmethod
    def player_count(self) -> int:
        pass

    @property
    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def _create_player(self, player_id: int) -> P:
        pass

    @property
    def players(self) -> list[P]:
        if self._players is None:
            self._players = [self._create_player(i) for i in range(self.player_count)]
        return self._players

    def update_display(self, surface: pygame.Surface):
        pass

    def process_event(self, event):
        pass


class GetOnlySimpleQueue:
    def __init__(self, queue: SimpleQueue):
        self.__queue = queue

    def get(self):
        while not self.__queue.empty():
            yield self.__queue.get_nowait()


class RendererEvent(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Renderer:
    def __init__(self):
        self.__game: Game = None
        self.__event = SimpleQueue()
        self.width: int = 500
        self.height: int = 600
        pygame.init()

    @property
    def game(self) -> Game:
        return self.__game

    @game.setter
    def game(self, value: Game):
        self.__game = value
        # Update Caption
        pygame.display.set_caption(self.__game.name)

    @property
    def event(self) -> GetOnlySimpleQueue:
        return GetOnlySimpleQueue(self.__event)

    def update(self):
        if self.__game is None:
            return

        # Handler UI Events
        events: list[Event] = pygame.event.get()
        for event in events:
            try:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.__event.put(RendererEvent.UP)
                    if event.key == pygame.K_RIGHT:
                        self.__event.put(RendererEvent.RIGHT)
                    if event.key == pygame.K_DOWN:
                        self.__event.put(RendererEvent.DOWN)
                    if event.key == pygame.K_LEFT:
                        self.__event.put(RendererEvent.LEFT)
            except Exception as e:
                print(str(e))

        display = pygame.display.set_mode((self.width, self.height))
        self.__game.update_display(display)
        pygame.display.update(display.get_rect())

        # Update UI
