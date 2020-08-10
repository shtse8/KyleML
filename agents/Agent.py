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


class Agent(object):
    def __init__(self, env, **kwargs) -> None:
        # Model
        self.env = env
        
        # Trainning
        self.target_epochs = kwargs.get('target_epochs', 1000)
        self.epochs = 0
        self.episodes = 0
        self.target_episodes = kwargs.get('episodes', 500)
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.highest_rewards = 0
        
        # History
        self.rewardHistory = collections.deque(maxlen=self.target_episodes)
        self.lossHistory = collections.deque(maxlen=self.target_episodes)
        self.bestReward = -np.Infinity
        
        self.epochStartTime = 0

    def beginEpoch(self):
        self.epochs += 1
        self.episodes = 0
        self.steps = 0
        self.rewardHistory.clear()
        self.lossHistory.clear()
        self.epochStartTime = time.perf_counter()
        return self.epochs <= self.target_epochs
    
    def endEpoch(self):
        bestReward = np.max(self.rewardHistory)
        if bestReward > self.bestReward:
            self.bestReward = bestReward
        self.save()
        print(f"")
    
    def printSummary(self):
        pass
       
    def get_action(self, state):
        raise NotImplementedError()
    
    def commit(self, transition: Transition):
        self.steps += 1
        self.total_rewards += transition.reward
        # self.update()
    
    
    def train(self):
        while self.beginEpoch():
            while self.beginEpisode():
                state = self.env.reset()
                done = False
                while not done:
                    action = self.get_action(state)
                    nextState, reward, done = self.env.takeAction(action)
                    self.commit(Transition(state, action, reward, nextState, done))
                    state = nextState
                self.endEpisode()
            self.endEpoch()
   
    def beginEpisode(self):
        self.episodes += 1
        self.steps = 0
        self.total_rewards = 0
        self.episode_start_time = time.perf_counter()
        return self.episodes <= self.target_episodes
        
    def endEpisode(self):
        self.rewardHistory.append(self.total_rewards)
        self.update()
      
    def update(self):
        duration = time.perf_counter() - self.epochStartTime
        avgLoss = np.mean(self.lossHistory) if len(self.lossHistory) > 0 else math.nan
        bestReward = np.max(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        avgReward = np.mean(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        
        print(f'Epoch #{self.epochs} {self.episodes:>5}/{self.target_episodes} | Loss: {avgLoss:8.4f} | Rewards: {self.total_rewards:>5} (Best: {bestReward:>5}, AVG: {avgReward:>5.2f}) | steps: {self.steps:>4} | Time: {duration: >5.2f}', end = "\r")
    
    def learn(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()
        
    def load(self):
        raise NotImplementedError()
                