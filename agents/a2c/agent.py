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
from memories.OnPolicy import OnPolicy, Transition
tf.compat.v1.disable_eager_execution()
from policies.Policy import Greedy, GaussianEpsGreedy

class Agent(object):
    def __init__(self, env, **kwargs):
        self.training = True
        self.nsteps = 1
        
        # History
        self.rewardHistory = collections.deque(maxlen=50)
        
        # Model
        self.env = env
        # Trainning
        
        self.episodes = 0
        self.target_episodes = kwargs.get('episodes', 10000)
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.highest_rewards = 0
        self.actor_weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + "_actor.h5")
        self.critic_weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + "_critic.h5")
        self.update_target_every = kwargs.get('update_target_every', 100)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.gamma = kwargs.get('gamma', 0.995)
        self.value_loss = kwargs.get('value_loss', 0.5)
        self.entropy_loss = kwargs.get('entropy_loss', 0.01)
        
        # Exploration
        self.epsilon_max = kwargs.get('epsilon_max', 1.00)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_phase_size = kwargs.get('epsilon_phase_size', 0.01)
        self.epsilon = self.epsilon_max
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_episodes * self.epsilon_phase_size))
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 10000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        
        self.train_overall_loss = []
        self.memory = collections.deque(maxlen=self.memory_size)
        
        self.actor = self.getActor()
        self.critic = self.getCritic()
        
        
    def printSummary(self):
        print(self.actor.summary())
        print(self.critic.summary())
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            
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
            
    
    def commit_memory(self, transition, instance=0):
        self.steps += 1
        self.total_rewards += transition.reward
        self.memory.append(transition)
    
    def train(self):
        while self.beginEpisode():
            state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                nextState, reward, done = self.env.takeAction(action)
                # print(state, "=>", nextState)
                self.commit_memory(Transition(
                    state = state, 
                    action = action, 
                    reward = reward,  
                    nextState = nextState if done else None))
                state = nextState
            self.endEpisode()
   
    def beginEpisode(self):
        self.episodes += 1
        self.steps = 0
        self.total_rewards = 0
        self.episode_start_time = time.perf_counter()
        return self.episodes <= self.target_episodes
        
    def endEpisode(self):
        info = self.learn()
        if self.total_rewards > self.highest_rewards:
            self.highest_rewards = self.total_rewards
        self.rewardHistory.append(self.total_rewards)
        duration = time.perf_counter() - self.episode_start_time
        print(f'Episode {self.episodes:>5}/{self.target_episodes} | Loss: {info["loss"]:8.4f} | Rewards: {self.total_rewards:>5} (Max: {self.highest_rewards:>5}, AVG: {np.mean(self.rewardHistory):>5.2f}) | steps: {self.steps:>4} | Epsilon: {self.epsilon:>4.2f} | Time: {duration: >5.2f}')
        self.update_epsilon()
        return {
            'loss': info['loss'],
            'duration': duration
        }
      
    def learn(self):
        
        tic = time.perf_counter()
        loss = 0
        
        states = [x.state for x in self.memory]
        rewards = [float(x.reward) for x in self.memory]
        actions = [x.action for x in self.memory]
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
            print("Weights loaded.")
        except:
            print("Failed to load.")
    
                