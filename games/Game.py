class Game(object):
    def __init__(self):
        pass

    def reset(self):
        return self.getState()

    def getState(self):
        return []

    def takeAction(self):
        return self.getState(), self.getReward(), self.getDone()

    def getReward(self):
        return 0

    def isDone(self):
        return False
        
    def getNew(self):
        return self.__class__()