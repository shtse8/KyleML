import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

class Game(object):
    def __init__(self):
        self.rendered = False

    def reset(self):
        return self.getState()

    def getState(self):
        return []

    def takeAction(self, action):
        self.update()
        return self.getState(), self.getReward(), self.getDone()

    def getReward(self):
        return 0

    def isDone(self):
        return False
        
    def getNew(self):
        return self.__class__()

    def render(self):
        pass

    def update(self):
        pass

