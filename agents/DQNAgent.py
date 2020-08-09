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
from memories.SimpleMemory import SimpleMemory
from memories.Transition import Transition
from .Agent import Agent

class DQNAgent(Agent):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Trainning
        self.weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + ".h5")
        self.update_target_every = kwargs.get('update_target_every', 20)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
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
        
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = Memory(self.memory_size)
        self.stmemory = SimpleMemory(self.memory_size)
        
        # Prediction model (the main Model)
        self.model = self.build_model()
        
        # Target model
        self.model_target = self.build_model()
        self.model_target.set_weights(self.model.get_weights())
        
        self.target_update_counter = 0

    def beginEpoch(self):
        self.stmemory.clear()
        self.ltmemory = Memory(self.memory_size)
        return super().beginEpoch()
        
    def printSummary(self):
        print(self.model.summary())

    def commit(self, transition: Transition):
        self.stmemory.add(transition)
        super().commit(transition)
        
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
                x = Conv2D(filters=32, kernel_size=1, strides=1, activation='relu')(x)
                # x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Conv2D(filters=32, kernel_size=1, strides=1, activation='relu')(x)
                # x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Conv2D(filters=64, kernel_size=1, strides=1, activation='relu')(x)
                # x = MaxPooling2D(pool_size=(2, 2))(x)
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
    
    def endEpisode(self):
        states, _, _, nextStates, _ = self.stmemory.get()
        # print(states.shape, nextStates.shape)
        # states, _, _, nextStates, _ = zip(*self.stmemory)
        # print(states, nextStates.shape)
        results = self.model.predict(np.concatenate((states, nextStates), axis=0))
        current_qs_list = results[0:len(states)]
        next_qs_list = results[len(states):]
        
        target_qs_list = self.model_target.predict(nextStates)
        for i, t in enumerate(self.stmemory):
            new_q = t.reward
            if not t.done:
                max_target_a = np.argmax(next_qs_list[i])
                # print(next_qs_list[i])
                max_target_q = target_qs_list[i][max_target_a] #Double DQN
                # max_target_q = np.amax(target_qs_list[i])
                new_q += self.discount_factor * max_target_q
                
            td_error = np.abs(new_q - current_qs_list[i][t.action])
            
            # frame = collections.namedtuple('frame', ['state', 'action', 'reward', 'done', 'next_state'])
            # print(td_error)
            self.ltmemory.add(td_error, t) 

            
        self.stmemory.clear()
        
        self.learn()
        self.update_epsilon()
        
        super().endEpisode()
      
    def learn(self):
        
        loss = 0
        
        # batch_size = min(len(self.stmemory), self.minibatch_size)
        # batch = random.sample(self.stmemory, batch_size)
        batch, idxs, is_weight = self.ltmemory.sample(self.minibatch_size)
        # print(batch, is_weight)
        
        states = np.array([x.state for x in batch])
        # print(np.array(states[:,1].tolist()))
        # current_qs_list = self.prediction_predict(states)
        
        nextStates = np.array([x.nextState for x in batch])
        target_qs_list = self.model_target.predict(nextStates)
        
        
        # next_qs_list = self.prediction_predict(nextStates) #Double DQN
        results = self.model.predict(np.concatenate((states, nextStates), axis=0))
        current_qs_list = results[0:len(states)]
        next_qs_list = results[len(states):]
        # print(results, len(results), current_qs_list, len(current_qs_list), next_qs_list)
        
        for i, t in enumerate(batch):
            new_q = t.reward
            if not t.done:
                max_target_a = np.argmax(next_qs_list[i])
                # print(next_qs_list[i])
                max_target_q = target_qs_list[i][max_target_a] #Double DQN
                # max_target_q = np.amax(target_qs_list[i])
                new_q += self.discount_factor * max_target_q
            
            # print(transition['action'], current_qs_list[i][transition['action']], "=>", new_q)
            error = np.abs(new_q - current_qs_list[i][t.action])
            current_qs_list[i][t.action] = new_q
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
                