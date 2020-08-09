import os
import tensorflow as tf
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
# from memories.Memory import Memory
from memories.OnPolicy import OnPolicy
from memories.Transition import Transition
from memories.SimpleMemory import SimpleMemory
from policies.Policy import Greedy, GaussianEpsGreedy
from .Agent import Agent

class A2CAgent(Agent):
    def __init__(self, env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        
        self.training = True
        self.nsteps = 1
        
        # History
        self.actor_weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + "_actor.h5")
        self.critic_weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + "_critic.h5")
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.gamma = kwargs.get('gamma', 0.995)
        self.value_loss = kwargs.get('value_loss', 0.5)
        self.entropy_loss = kwargs.get('entropy_loss', 0.01)
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 10000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        
        self.train_overall_loss = []
        self.memory = SimpleMemory(self.memory_size)
        
        self.actor = self.getActor()
        self.critic = self.getCritic()
        
        
    def printSummary(self):
        print(self.actor.summary())
        print(self.critic.summary())
    
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * self.gamma + reward[i]
            discounted_r[i] = running_add
        # print(discounted_r, reward)
        discounted_r -= np.mean(discounted_r) # normalizing the result
        # discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r


    def getActor(self, training = True):
        # Define two input layers
    
        state_input = Input(self.env.observationSpace, name="state_input")
        x = state_input
        x = Flatten()(state_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.5)(x, training = training)
        
        output = Dense(self.env.actionSpace, activation='softmax', kernel_initializer='he_uniform')(x) # Actor (Policy Network)
        model = Model(inputs=[state_input], outputs=output)
        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
        return model
        
    def getCritic(self, training = True):
    
        state_input = Input(self.env.observationSpace, name="state_input")
        x = state_input
        x = Flatten()(state_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        # x = Dropout(0.5)(x, training = training)
        
        output = Dense(1, kernel_initializer='he_uniform')(x) # Actor (Policy Network)
        model = Model(inputs=[state_input], outputs=output)
        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        return model
    
    def get_action(self, state):
        # print(np.array(state))
        prediction = self.actor.predict([np.array([state])])[0]
        # print(prediction)
        action = np.random.choice(self.env.actionSpace, p=prediction)
        # print(action)
        return action
            
    
    def commit(self, transition: Transition):
        self.memory.add(transition)
        super().commit(transition)
        
    def endEpisode(self):
        self.learn()
        super().endEpisode()
      
    def learn(self):
        
        tic = time.perf_counter()
        loss = 0
        
        states, actions, rewards, _, _ = self.memory.get()
        
        # Compute discounted rewards
        discounted_r = self.discount_rewards(rewards)
        # print(discounted_r)

        values = self.critic.predict([states])[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        # print(advantages)
        result = self.actor.fit([states], actions, sample_weight=advantages, epochs=1, verbose=0)
        loss += result.history['loss'][0]
        reuslt = self.critic.fit([states], discounted_r, epochs=1, verbose=0)
        loss += result.history['loss'][0]
        # reset training memory
        self.memory.clear()
        
            
        self.save()
            
                
        toc = time.perf_counter()
        return {
            'loss': loss,
            'duration': toc - tic
        }

    def save(self):
        try:
            self.actor.save_weights(self.actor_weights_path)
            self.critic.save_weights(self.critic_weights_path)
            # print("Saved Weights.")
        except:
            print("Failed to save.")
        
    def load(self):
        try:
            self.actor.load_weights(self.actor_weights_path)
            self.critic.load_weights(self.critic_weights_path)
            # print("Weights loaded.")
        except:
            print("Failed to load.")
    
                