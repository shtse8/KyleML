class Game(object):
    def __init__(self):
        self.rendered = False

    # Game methods
    def getPlayer(self, playerId: int):
        return GamePlayer(self, playerId)

    def getPlayerCount(self):
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    def isDone(self) -> bool:
        raise NotImplementedError()

    # Player Methods
    def canStep(self, playerId):
        raise NotImplementedError()

    def getMask(self, playerId: int):
        raise NotImplementedError()

    def getState(self, playerId: int):
        raise NotImplementedError()

    def _step(self, playerId: int, action) -> None:
        raise NotImplementedError()

    def step(self, playerId: int, action) -> tuple:
        self._step(action)
        self.update()
        return self.getState(), self.getReward(), self.isDone()

    def getReward(self) -> float:
        raise NotImplementedError()

    # UI Methods
    def render(self) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        raise NotImplementedError()


class GamePlayer:
    def __init__(self, game: Game, playerId: int):
        self.game = game
        self.playerId = playerId

    def getState(self):
        return self.game.getState(self.playerId)

    def canStep(self):
        return self.game.canStep(self.playerId)

    def getMask(self, state):
        return self.game.mask(self.playerId, state)

    def step(self, action) -> tuple:
        return self.game.step(self.playerId, action)

