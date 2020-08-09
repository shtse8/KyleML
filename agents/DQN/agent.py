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
from memories.Memory import Memory
tf.compat.v1.disable_eager_execution()


class Agent(object):
    def __init__(self, env, **kwargs):
    
        
        # Model
        self.env = env
        
        # Trainning
        self.target_epochs = kwargs.get('target_epochs', 1000)
        self.epochs = 0
        self.episodes = 0
        self.target_episodes = kwargs.get('episodes', 200)
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.highest_rewards = 0
        self.weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + ".h5")
        self.update_target_every = kwargs.get('update_target_every', 20)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.discount_factor = kwargs.get('discount_factor', 0.995)
        
        
        # Exploration
        self.epsilon_max = kwargs.get('epsilon_max', 1.00)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_phase_size = kwargs.get('epsilon_phase_size', 0.5)
        self.epsilon = self.epsilon_max
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_episodes * self.epsilon_phase_size))
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 10000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        
        self.train_overall_loss = []
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = Memory(self.memory_size)
        self.stmemory = collections.deque(maxlen=self.memory_size)
        
        # Prediction model (the main Model)
        self.model = self.build_model()
        
        # Target model
        self.model_target = self.build_model()
        self.model_target.set_weights(self.model.get_weights())
        
        self.target_update_counter = 0
        
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
        self.epsilon = self.epsilon_max
        self.stmemory.clear()
        self.ltmemory = Memory(self.memory_size)
        return self.epochs <= self.target_epochs
    
    def endEpoch(self):
        bestReward = np.max(self.rewardHistory)
        if bestReward > self.bestReward:
            self.bestReward = bestReward
        self.save()
        print(f"")
    
    def printSummary(self):
        print(self.model.summary())
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            # For exploration
            action = random.randint(0, self.env.actionSpace - 1)
        else:
            prediction = self.model.predict(np.array([state]))
            action = np.argmax(prediction[0])
        
        # print(f'[{game.counter}-{game.record}]', game.steps, "taking action:", action == 2 and "L" or action == 1 and "R" or "S", reward, game.score, prediction[0])
        return action
    
    def build_model(self, training = True):
        # Define two input layers
        
        if tf.config.list_physical_devices('gpu'):
          strategy = tf.distribute.MirroredStrategy()
        else:  # use default strategy
          strategy = tf.distribute.get_strategy() 

        with strategy.scope():
            
            state_input = Input(self.env.observationSpace, name="state_input")
            x = state_input
            if len(self.env.observationSpace) == 3:
                # normalized = Lambda(lambda x: x / 4.0)(obseration_input)
                x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Conv2D(filters=32, kernel_size=1, strides=1, activation='relu')(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                # x = Dropout(0.25)(x)
                # x = Flatten()(x)
                # state_input = Input((4), name="state_input")
                # concatenated = Concatenate()([flat_layer, state_input])
            # else 

            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            # x = Dense(64, activation='relu')(x)
            # x = Dense(32, activation='relu')(x)
            # x = Dropout(0.5)(x, training = training)
            
            # Dueling DQN
            state_value = Dense(1, kernel_initializer='he_uniform')(x)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=self.env.actionSpace)(state_value)

            action_advantage = Dense(self.env.actionSpace, kernel_initializer='he_uniform')(x)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=self.env.actionSpace)(action_advantage)

            output = Add()([state_value, action_advantage])
            
            # value = Dense(self.env.actionSpace, activation='linear')(hidden)
            # a = Dense(self.env.actionSpace, activation='linear')(hidden)
            # meam = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
            # advantage = Subtract()([a, meam])
            # output = Add()([value, advantage])

            # output = Dense(self.env.actionSpace, activation='softmax')(q)
            # output = Dense(3)(hidden3)
            model = Model(inputs=[state_input], outputs=output)
        # opt = Adam(lr=LEARNING_RATE, clipnorm=0.1)
        opt = Adam(lr=self.learning_rate)
        # opt = RMSprop(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        
        
        # sys.exit(1)
        return model
    
    def commit_memory(self, state, action, reward, done, next_state):
        self.steps += 1
        self.total_rewards += reward
        
        self.stmemory.append((
            state,
            action, 
            reward, 
            next_state,
            done
        ))
        
        self.update()
    
    def train(self):
        while self.beginEpoch():
            while self.beginEpisode():
                state = self.env.reset()
                done = False
                while not done:
                    action = self.get_action(state)
                    nextState, reward, done = self.env.takeAction(action)
                    # print(state, "=>", nextState)
                    self.commit_memory(state, action, reward, done, nextState)
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
    
        states = np.array([x[0] for x in self.stmemory])
        next_states = np.array([x[3] for x in self.stmemory])
        # states, _, _, next_states, _ = zip(*self.stmemory)
        # print(states, next_states.shape)
        results = self.model.predict(np.concatenate((states, next_states), axis=0))
        current_qs_list = results[0:len(states)]
        next_qs_list = results[len(states):]
        
        target_qs_list = self.model_target.predict(next_states)
        for i, (state, action, reward, next_state, done) in enumerate(self.stmemory):
            new_q = reward
            if not done:
                max_target_a = np.argmax(next_qs_list[i])
                # print(next_qs_list[i])
                max_target_q = target_qs_list[i][max_target_a] #Double DQN
                # max_target_q = np.amax(target_qs_list[i])
                new_q += self.discount_factor * max_target_q
                
            td_error = np.abs(new_q - current_qs_list[i][action])
            
            # frame = collections.namedtuple('frame', ['state', 'action', 'reward', 'done', 'next_state'])
            # print(td_error)
            self.ltmemory.add(td_error, (
                state,
                action, 
                reward, 
                next_state,
                done
            )) 

            
        self.stmemory.clear()
        
        self.learn()
        self.rewardHistory.append(self.total_rewards)
        self.update_epsilon()
        
        self.update()
      
    def update(self):
        duration = time.perf_counter() - self.epochStartTime
        avgLoss = np.mean(self.lossHistory) if len(self.lossHistory) > 0 else math.nan
        bestReward = np.max(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        avgReward = np.mean(self.rewardHistory) if len(self.rewardHistory) > 0 else math.nan
        
        print(f'Epoch #{self.epochs} {self.episodes:>5}/{self.target_episodes} | Loss: {avgLoss:8.4f} | Rewards: {self.total_rewards:>5} (Best: {bestReward:>5}, AVG: {avgReward:>5.2f}) | steps: {self.steps:>4} | Epsilon: {self.epsilon:>4.2f} | Time: {duration: >5.2f}', end = "\r")
    
    def learn(self):
        
        loss = 0
        
        # batch_size = min(len(self.stmemory), self.minibatch_size)
        # batch = random.sample(self.stmemory, batch_size)
        batch, idxs, is_weight = self.ltmemory.sample(self.minibatch_size)
        # print(batch, is_weight)
        
        states = np.array([x[0] for x in batch])
        # print(np.array(states[:,1].tolist()))
        # current_qs_list = self.prediction_predict(states)
        
        next_states = np.array([x[3] for x in batch])
        target_qs_list = self.model_target.predict(next_states)
        
        
        # next_qs_list = self.prediction_predict(next_states) #Double DQN
        results = self.model.predict(np.concatenate((states, next_states), axis=0))
        current_qs_list = results[0:len(states)]
        next_qs_list = results[len(states):]
        # print(results, len(results), current_qs_list, len(current_qs_list), next_qs_list)
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            new_q = reward
            if not done:
                max_target_a = np.argmax(next_qs_list[i])
                # print(next_qs_list[i])
                max_target_q = target_qs_list[i][max_target_a] #Double DQN
                # max_target_q = np.amax(target_qs_list[i])
                new_q += self.discount_factor * max_target_q
            
            # print(transition['action'], current_qs_list[i][transition['action']], "=>", new_q)
            error = np.abs(new_q - current_qs_list[i][action])
            current_qs_list[i][action] = new_q
            self.ltmemory.update(idxs[i], error)
            
        fit = self.model.fit(
            # fit_generator(states, current_qs_list, KERAS_BATCH_SIZE),
            states,
            current_qs_list,
            sample_weight=is_weight,
            epochs=1,
            verbose=0,
            validation_split=0,
            # batch_size=batch_size,
            steps_per_epoch=None,
            # batch_size=32,
            workers = 32,
            use_multiprocessing=False,
        )
        
        loss += fit.history['loss'][0]
        
        
        # Update target model counter every episode
        self.target_update_counter += 1
        # self.ltmemory.clear()
        
        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter >= self.update_target_every:
            self.updateTarget()
            
            
        self.lossHistory.append(loss)

    def save(self):
        try:
            self.model.save_weights(self.weights_path)
            # print("Saved Weights.")
        except:
            print("Failed to save.")
        
    def load(self):
        try:
            self.model.load_weights(self.weights_path)
            print("Weights loaded.")
        except:
            print("Failed to load.")
    
    def updateTarget(self):
        # print("Target is updated.")
        self.model_target.set_weights(self.model.get_weights())
        self.target_update_counter = 0
                