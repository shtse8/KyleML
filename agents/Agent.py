import os
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
from pathlib import Path
import matplotlib.pyplot as plt
from memories.Transition import Transition
from utils.errors import InvalidAction
from enum import Enum
import torch

class Phrase(Enum):
    train = 1
    test = 2
    play = 3

class Agent(object):
    def __init__(self, env, **kwargs) -> None:
        self.name = ""

        # Model
        self.env = env
        
        # Trainning
        self.target_epochs = kwargs.get('target_epochs', 10000)
        self.target_trains = kwargs.get('target_trains', 10000)
        self.target_tests = kwargs.get('target_tests', 100)
        
        self.target_episodes = 0
        
        # self.phrases = [Phrase.train, Phrase.test]
        self.phrases = []
        # self.phrases = [Phrase.play]
        self.phraseIndex = 0
        self.phrase = ""
        self.epochs = 0
        self.episodes = 0
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.total_loss = 0
        self.highest_rewards = 0
        self.invalidMoves = 0
        # History
        self.rewardHistory = collections.deque(maxlen=max(self.target_trains, self.target_tests))
        self.lossHistory = collections.deque(maxlen=max(self.target_trains, self.target_tests))
        self.bestReward = -np.Infinity
        
        self.startTime = 0

        # Save And Load
        self.weightPath = kwargs.get('weightPath', "./weights/")
        self.models = []

        # plt.ion()
        plt.ylabel("Loss")
        plt.xlabel("Episode")
        # plt.show(block=False)


    def addModels(self, model) -> None:
        self.models.append(model)

    def beginEpoch(self) -> None:
        self.epochs += 1
        self.phraseIndex = 0
        return self.epochs <= self.target_epochs
    
    def endEpoch(self) -> None:
        pass

    def getPrediction(self, state):
        raise NotImplementedError()

    def getAction(self, state, actionMask = None) -> int:
        raise NotImplementedError()
    
    def commit(self, transition: Transition) -> None:
        self.total_rewards += transition.reward
        # self.update()
    
    def applyMask(self, prediction, mask, replacedValue = 0):
        return np.array([prediction[i] if mask else replacedValue for i, mask in enumerate(mask)])

    def run(self, train = True) -> None:
        self.phrases = [Phrase.train] if train else [Phrase.play]
        while self.beginEpoch():
            while self.beginPhrase():
                while self.beginEpisode():
                    state = self.env.reset()
                    done = False
                    while not done:
                        reward = 0
                        nextState = state
                        actionMask = np.ones(self.env.actionSpace)
                        prediction = self.getPrediction(state)
                        while True:
                            try:
                                action = self.getAction(prediction, actionMask)
                                nextState, reward, done = self.env.takeAction(action)
                                self.commit(Transition(state, action, reward, nextState, done))
                                break
                            except InvalidAction:
                                actionMask[action] = 0
                                self.invalidMoves += 1
                                # print(actionMask)
                            # finally:
                        state = nextState
                        if self.isPlaying():
                            time.sleep(0.3)
                    self.endEpisode()
                self.endPhrase()
            self.endEpoch()
            
    def getPhrase(self) -> Phrase:
        return self.phrases[self.phraseIndex]
        
    def isTraining(self) -> bool:
        return self.getPhrase() == Phrase.train
        
    def isTesting(self) -> bool:
        return self.getPhrase() == Phrase.test
        
    def isPlaying(self) -> bool:
        return self.getPhrase() == Phrase.play
        
    def beginPhrase(self) -> None:
        if self.phraseIndex >= len(self.phrases):
            return False
        self.episodes = 0
        self.steps = 0
        self.invalidMoves = 0
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
    
    def endPhrase(self) -> None:
        if self.isTraining():
            self.save()

        self.phraseIndex += 1
        # plt.cla()
        plt.plot(self.lossHistory)
        plt.show()
        # plt.draw()
        # plt.pause(0.00001)
        print()
        
    def beginEpisode(self) -> None:
        self.episodes += 1
        self.total_rewards = 0
        self.total_loss = 0
        self.episode_start_time = time.perf_counter()
        return self.episodes <= self.target_episodes
        
    def endEpisode(self) -> None:
        self.lossHistory.append(self.total_loss)
        self.rewardHistory.append(self.total_rewards)
        # bestReward = np.max(self.rewardHistory)
        # if bestReward > self.bestReward:
            # self.bestReward = bestReward
        self.update()
      
    def update(self) -> None:
        duration = time.perf_counter() - self.startTime
        avgLoss = np.mean(self.lossHistory) if len(self.lossHistory) > 0 else math.nan
        bestReward = np.max(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        avgReward = np.mean(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        stdReward = np.std(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        progress = self.episodes / self.target_episodes
        invalidMovesPerEpisode = self.invalidMoves / self.episodes
        durationPerEpisode = duration /  self.episodes
        estimateDuration = self.target_episodes * durationPerEpisode
        print(f"Epoch #{self.epochs:>3} {self.getPhrase().name:5} {progress:>4.0%} | " + \
            f'Loss: {avgLoss:6.2f}/ep | Best: {bestReward:>5}, Avg: {avgReward:>5.2f}, Std: {stdReward:>5.2f} | Steps: {self.steps/duration:>7.2f}/s, {self.steps/self.episodes:>6.2f}/ep | Episodes: {1/durationPerEpisode:>6.2f}/s | Invalid: {invalidMovesPerEpisode: >6.2f} | Time: {duration: >4.2f}s > {estimateDuration: >5.2f}s', end = "\b\r")
    
    def learn(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        try:
            for model in self.models:
                path = self.getModelPath(model, True)
                torch.save(model.state_dict(), path)
            # print("Saved Weights.")
        except Exception as e:
            print("Failed to save.", e)
        
    def load(self) -> None:
        try:
            for model in self.models:
                path = self.getModelPath(model)
                print("Loading from path: ", path)
                model.load_state_dict(torch.load(path))
            print("Weights loaded.")
        except Exception as e:
            print("Failed to load.", e)
                

    def getModelPath(self, model, makeDir = False):
        path = os.path.join(os.path.dirname(__main__.__file__), self.weightPath, self.name, self.env.name, model.name + ".h5")
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        return path