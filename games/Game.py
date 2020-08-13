import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
COLOR_SWITCHER = {
    1: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (244, 149, 99),
    32: (245, 121, 77),
    64: (245, 93, 55),
    128: (238, 232, 99),
    256: (237, 176, 77),
    512: (236, 176, 77),
    1024: (235, 148, 55),
    2048: (234, 120, 33),
}

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
        if self.render:
            return
        self.rendered = True
        self.width = 500
        self.height = 600
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.name)
        pygame.init()

    def update(self):
        if not self.rendered:
            return
            
        pygame.event.get()
        state = self.getState()
        font = pygame.font.Font(pygame.font.get_default_font(), 36)
        scoreHeight = 100
        
        score_rect = pygame.Rect(0, 0, self.width, scoreHeight)
        pygame.draw.rect(self.display, (0, 0, 0), score_rect)
        text = font.render(str(self.game.score), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = score_rect.center
        self.display.blit(text, text_rect)
        
        for y, row in enumerate(state):
            for x, cell in enumerate(row):
                block_size = ((self.height - scoreHeight) / self.observationSpace[0], self.width / self.observationSpace[1])
                block_rect = pygame.Rect(x * block_size[1], scoreHeight + y * block_size[0], block_size[1], block_size[0])
                pygame.draw.rect(self.display, COLOR_SWITCHER[2 ** cell], block_rect)
                if not cell == 0:
                    text = font.render(str(2 ** cell), True, (0, 0, 0))
                    text_rect = text.get_rect()
                    text_rect.center = block_rect.center
                    self.display.blit(text, text_rect)
        pygame.display.update()

