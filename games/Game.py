class Game(object):
    def __init__(self):
        self.rendered = False

    def reset(self):
        return self.getState()

    def getState(self):
        return []

    def takeAction(self, action) -> tuple:
        self.update()
        return self.getState(), self.getReward(), self.getDone()

    def getReward(self) -> float:
        return 0

    def isDone(self) -> bool:
        return False

    def getNew(self):
        return self.__class__()

    def render(self) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        raise NotImplementedError()

