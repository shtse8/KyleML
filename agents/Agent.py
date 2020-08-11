import os
import tensorflow as tf
from keras.callbacks import Callback
from keras.optimizers import Adam, RMSprop
from keras.utils import Sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Concatenate, Input, Flatten, Conv2D, Lambda, MaxPooling2D, Subtract, Add
from keras.utils import to_categorical
import keras.backend as K
from multiprocessing import Pool, TimeoutError
import __main__
import random
import numpy as np
import pandas as pd
from operator import add
import sys
import collections
import math
import time
from memories.Transition import Transition
tf.compat.v1.disable_eager_execution()
from enum import Enum

class Phrase(Enum):
    train = 1
    test = 2
    play = 3

class Agent(object):
    def __init__(self, env, **kwargs) -> None:
        # Model
        self.env = env
        
        # Trainning
        self.target_epochs = kwargs.get('target_epochs', 10000)
        self.target_trains = kwargs.get('target_trains', 10000)
        self.target_tests = kwargs.get('target_tests', 100)
        
        self.target_episodes = 0
        # self.phrases = [Phrase.train, Phrase.test]
        # self.phrases = [Phrase.train]
        self.phrases = [Phrase.play]
        self.phraseIndex = 0
        self.phrase = ""
        self.epochs = 0
        self.episodes = 0
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.highest_rewards = 0
        
        # History
        self.rewardHistory = collections.deque(maxlen=max(self.target_trains, self.target_tests))
        self.lossHistory = collections.deque(maxlen=max(self.target_trains, self.target_tests))
        self.bestReward = -np.Infinity
        
        self.startTime = 0

    def beginEpoch(self):
        self.epochs += 1
        self.phraseIndex = 0
        return self.epochs <= self.target_epochs
    
    def endEpoch(self):
        self.save()
    
    def printSummary(self):
        pass
       
    def getAction(self, state):
        raise NotImplementedError()
    
    def commit(self, transition: Transition):
        self.steps += 1
        self.total_rewards += transition.reward
        # self.update()
    
    def train(self):
        while self.beginEpoch():
            while self.beginPhrase():
                while self.beginEpisode():
                    state = self.env.reset()
                    done = False
                    while not done:
                        action = self.getAction(state)
                        nextState, reward, done = self.env.takeAction(action)
                        self.commit(Transition(state, action, reward, nextState, done))
                        state = nextState
                        if self.isPlaying():
                            time.sleep(0.3)
                    self.endEpisode()
                self.endPhrase()
            self.endEpoch()
            
    def getPhrase(self):
        return self.phrases[self.phraseIndex]
        
    def isTraining(self):
        return self.getPhrase() == Phrase.train
        
    def isTesting(self):
        return self.getPhrase() == Phrase.test
        
    def isPlaying(self):
        return self.getPhrase() == Phrase.play
        
    def beginPhrase(self):
        if self.phraseIndex >= len(self.phrases):
            return False
        self.episodes = 0
        self.steps = 0
        self.rewardHistory.clear()
        self.lossHistory.clear()
        self.startTime = time.perf_counter()
        if self.isTraining():
            self.target_episodes = self.target_trains
        elif self.isTesting():
            self.target_episodes = self.target_tests
        elif self.isPlaying():
            self.target_episodes = 1
        return True
    
    def endPhrase(self):
        self.phraseIndex += 1
        print()
        
    def beginEpisode(self):
        self.episodes += 1
        self.total_rewards = 0
        self.episode_start_time = time.perf_counter()
        return self.episodes <= self.target_episodes
        
    def endEpisode(self):
        self.rewardHistory.append(self.total_rewards)
        # bestReward = np.max(self.rewardHistory)
        # if bestReward > self.bestReward:
            # self.bestReward = bestReward
        self.update()
      
    def update(self):
        duration = time.perf_counter() - self.startTime
        avgLoss = np.mean(self.lossHistory) if len(self.lossHistory) > 0 else math.nan
        bestReward = np.max(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        avgReward = np.mean(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        progress = self.episodes / self.target_episodes
        print(f'Epoch #{self.epochs:>3} {self.getPhrase().name:5} {progress:>4.0%} | Loss: {avgLoss:8.4f}/ep | Best: {bestReward:>5}, AVG: {avgReward:>5.2f} | steps: {self.steps/duration:>7.2f}/s | Time: {duration: >5.2f}s', end = "\r")
    
    def learn(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
        
    def load(self):
        raise NotImplementedError()
                