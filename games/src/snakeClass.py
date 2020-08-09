import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
#import tensorflow as tf    
#tf.get_logger().setLevel('ERROR')

import pygame
import math
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from collections import deque
from random import randint
from keras.utils import to_categorical
import atexit
import signal
import sys
from operator import add

PIXEL_SIZE = 20



class Game:
    def __init__(self, width, height, render = False):
        self.width = width
        self.height = height
        self.is_display = render
        
        self.record = 0
        self.counter = 0
        self.last_update = 0
        self.resources = {}
        self.bgImagePath = 'img/background.png'
        self.playerImagePath = 'img/snakeBody.png'
        self.foodImagePath = 'img/food2.png'
        self.isResourceLoaded = False

    def start(self):
        self.crash = False
        self.end = False
        self.player = Player(self)
        self.food = Food(self)
        self.steps = 0
        self.temp_steps = 0
        self.score = 0
        self.counter += 1
        
    def eat(self):
        if self.player.x == self.food.x and self.player.y == self.food.y:
            self.food.new()
            self.player.eaten = True
            self.score = self.score + 1
            if self.score > self.record:
                self.record = self.score
    def load_resource(self):
    
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption('SnakeGen')
        self.gameDisplay = pygame.display.set_mode((self.width * PIXEL_SIZE, self.height * PIXEL_SIZE + 60))
        
        self.resources['playerImage'] = pygame.image.load(self.playerImagePath)
        self.resources['foodImage'] = pygame.image.load(self.foodImagePath)
        self.resources['bgImage'] = pygame.image.load(self.bgImagePath)
        
        self.resources['myfont'] = pygame.font.SysFont('Segoe UI', 20)
        self.resources['myfont_bold'] = pygame.font.SysFont('Segoe UI', 20, True)
        self.resources['text_score'] = self.resources['myfont'].render('SCORE: ', True, (0, 0, 0))
        self.resources['text_score_number'] = self.resources['myfont'].render(str(self.score), True, (0, 0, 0))
        self.resources['text_highest'] = self.resources['myfont'].render('HIGHEST SCORE: ', True, (0, 0, 0))
        self.resources['text_highest_number'] = self.resources['myfont_bold'].render(str(self.record), True, (0, 0, 0))
        self.isResourceLoaded = True
        
    def display(self):
        if not self.is_display:
            return
        
        if time.perf_counter() - self.last_update < 1/30:
            return
        self.last_update = time.perf_counter()
        
        if not self.isResourceLoaded:
            self.load_resource()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
            
        # Clear
        self.gameDisplay.fill((255, 255, 255))
        
        # Draw UI
        self.gameDisplay.blit(self.resources['text_score'], (45, 440))
        self.gameDisplay.blit(self.resources['text_score_number'], (120, 440))
        self.gameDisplay.blit(self.resources['text_highest'], (190, 440))
        self.gameDisplay.blit(self.resources['text_highest_number'], (350, 440))
        self.gameDisplay.blit(self.resources['bgImage'], (10, 10))
        
        # Draw Player
        if self.crash == False:
            for i in range(self.player.food):
                x_temp, y_temp = self.player.position[len(self.player.position) - 1 - i]
                self.gameDisplay.blit(self.resources['playerImage'], (x_temp * PIXEL_SIZE, y_temp * PIXEL_SIZE))
                
        # Draw new food
        self.gameDisplay.blit(self.resources['foodImage'], (self.food.x * PIXEL_SIZE, self.food.y * PIXEL_SIZE))
        
        pygame.display.update()
        


class Player(object):
    def __init__(self, game):
        self.x = math.floor((game.width - 1) * 0.45)
        self.y = math.floor((game.height - 1) * 0.5)
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.x_change = 1
        self.y_change = 0

    def update_position(self, x, y):
        if self.position[-1][0] != x or self.position[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def do_move(self, move, game):
        move_array = [self.x_change, self.y_change]
        game.steps += 1
        game.temp_steps += 1
        
        if self.eaten:
            game.temp_steps = 0
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
            
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x += self.x_change
        self.y += self.y_change
        
        if self.x <= 0 or self.x >= game.width - 1 \
                or self.y <= 0 \
                or self.y >= game.height - 1 \
                or [self.x, self.y] in self.position:
            game.crash = True
            game.end = True
            
        if game.temp_steps >= 100:
            game.end = True
            
        game.eat()

        self.update_position(self.x, self.y)
        



class Food(object):
    def __init__(self, game):
        self.game = game
        self.new()
        
    def new(self):
        while True:
            self.x = randint(1, self.game.width - 2)
            self.y = randint(1, self.game.height - 2)
            if [self.x, self.y] not in self.game.player.position:
                break


