import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from Memory import Memory
tf.compat.v1.disable_eager_execution()

class predict_generator(Sequence):

    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, i):
        start = i * self.batch_size
        end = start + self.batch_size
        batch_x = [
            np.array(self.x[start:end,0].tolist()), 
            np.array(self.x[start:end,1].tolist())
        ]
        return batch_x
        
class fit_generator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, i):
        start = i * self.batch_size
        end = start + self.batch_size
        batch_x = [
            np.array(self.x[start:end,0].tolist()), 
            np.array(self.x[start:end,1].tolist())
        ]
        batch_y = np.array(self.y[start:end])

        return batch_x, batch_y
        

class Agent(object):
    def __init__(self, inputShape, outputShape, **kwargs):
    
        # History
        self.rewardHistory = collections.deque(maxlen=50)
        
        # Model
        self.inputShape = inputShape
        self.outputShape = outputShape
        
        # Trainning
        self.train = kwargs.get('train', True)
        self.episodes = 0
        self.target_episodes = kwargs.get('episodes', 10000)
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.highest_rewards = 0
        self.weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + ".h5")
        self.update_target_every = kwargs.get('update_target_every', 100)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.discount_factor = kwargs.get('discount_factor', 0.995)
        
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
        
        print(f'epsilon_decay = {self.epsilon_decay}')
        self.train_overall_loss = []
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = Memory(self.memory_size)
        self.stmemory = collections.deque(maxlen=self.memory_size)
        
        # Prediction model (the main Model)
        self.model_prediction = self.build_model()
        # self.model_prediction.make_test_function()
        # self.model_prediction.make_train_function()
        # self.model_prediction.make_predict_function()
        
        print(self.model_prediction.summary())
        
        if os.path.exists(self.weights_path):
            # Ask for clearing weights
            clear_weights = input("Clear weights? (y/n)")
            if clear_weights == "y":
                try:
                    os.remove(self.weights_path)
                except:
                    print("Failed to remove")
            else:
                try:
                    self.model_prediction.load_weights(self.weights_path)
                    print("weights loaded")
                except:
                    input("Failed to load weights. Continue?")
        
        # Target model
        self.model_target = self.build_model()
        self.model_target.set_weights(self.model_prediction.get_weights())
        # Build and compile the predict function on CPU/GPU first.
        # self.model_target.make_predict_function()
        # self.model_prediction.make_test_function()
        
        self.target_update_counter = 0

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_action(self, state):
        # if not self.train:
            # return self.get_predict_action(state)
        # return self.get_predict_action(state)
        
        
        
        # tau = 0.1
        # clip = (-500., 500.)
        # q_values = self.prediction_predict(np.array([state]))[0]
        # if np.random.uniform() < self.epsilon:
            # exp_values = np.exp(np.clip(q_values / tau, clip[0], clip[1]).astype(np.longdouble))
            # probs = exp_values / np.sum(exp_values)
            # action = np.random.choice(range(self.outputShape), p=probs)
        # else:
            # action = np.argmax(q_values)
            
        if np.random.uniform() < self.epsilon:
            # For exploration
            prediction = [None]
            action = random.randint(0, self.outputShape - 1)
        else:
            action, prediction = self.get_predict_action(state)
        
        # print(f'[{game.counter}-{game.record}]', game.steps, "taking action:", action == 2 and "L" or action == 1 and "R" or "S", reward, game.score, prediction[0])
        return action
    
    def get_predict_action(self, state):
        prediction = self.prediction_predict(np.array([state]))
        # predict action based on the old state
        action = np.argmax(prediction[0])
        return action, prediction
    
    def build_model(self, training = True):
        # Define two input layers
        
        if tf.config.list_physical_devices('gpu'):
          strategy = tf.distribute.MirroredStrategy()
        else:  # use default strategy
          strategy = tf.distribute.get_strategy() 

        with strategy.scope():
            
            # obseration_input = Input((22, 22, 1), name="obseration_input")
            # # normalized = Lambda(lambda x: x / 4.0)(obseration_input)
            # conv1_layer = Conv2D(filters=8, kernel_size=3, strides=1, activation='relu')(obseration_input)
            # pooling1_layer = MaxPooling2D(pool_size=(2, 2))(conv1_layer)
            # conv2_layer = Conv2D(filters=16, kernel_size=3, strides=1, activation='relu')(pooling1_layer)
            # pooling2_layer = MaxPooling2D(pool_size=(2, 2))(conv2_layer)
            # conv3_layer = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(pooling2_layer)
            # pooling3_layer = MaxPooling2D(pool_size=(2, 2))(conv3_layer)
            # # dropout1 = Dropout(0.25)(conv3_layer)
            # flat_layer = Flatten()(pooling3_layer)
            # state_input = Input((4), name="state_input")
            # concatenated = Concatenate()([flat_layer, state_input])

            state_input = Input(self.inputShape, name="state_input")
            x = state_input
            x = Flatten()(state_input)
            x = Dense(128, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            # x = Dropout(0.5)(x, training = training)
            
            # Dueling DQN
            state_value = Dense(1, kernel_initializer='he_uniform')(x)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=self.outputShape)(state_value)

            action_advantage = Dense(self.outputShape, kernel_initializer='he_uniform')(x)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=self.outputShape)(action_advantage)

            output = Add()([state_value, action_advantage])
            
            # value = Dense(self.outputShape, activation='linear')(hidden)
            # a = Dense(self.outputShape, activation='linear')(hidden)
            # meam = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
            # advantage = Subtract()([a, meam])
            # output = Add()([value, advantage])

            # output = Dense(self.outputShape, activation='softmax')(q)
            # output = Dense(3)(hidden3)
            model = Model(inputs=[state_input], outputs=output)
        # opt = Adam(lr=LEARNING_RATE, clipnorm=0.1)
        opt = Adam(lr=self.learning_rate)
        # opt = RMSprop(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=opt)
        
        
        # sys.exit(1)
        return model
    
    def prediction_predict(self, states):
        # return self.model_prediction.predict(predict_generator(states, KERAS_BATCH_SIZE), steps=None, workers=32, use_multiprocessing=False)
        return self.model_prediction.predict([
                np.array(states[:,0].tolist()), 
                # np.array(states[:,1].tolist())
            ], workers=32, use_multiprocessing=False)

    def target_predict(self, states):
        # return self.model_target.predict(predict_generator(states, KERAS_BATCH_SIZE), steps=None, workers=32, use_multiprocessing=False)
        return self.model_target.predict([
                np.array(states[:,0].tolist()), 
                # np.array(states[:,1].tolist())
            ], workers=32, use_multiprocessing=False)

        
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
    
    def begin_episode(self):
        self.episodes += 1
        self.steps = 0
        self.total_rewards = 0
        self.episode_start_time = time.perf_counter()
        return self.episodes <= self.target_episodes
        
    def end_episode(self):
        # if len(self.ltmemory) >= MIN_REPLAY_MEMORY_SIZE:
            # self.replay()
            
        current_states = np.array([x[0] for x in self.stmemory])
        # print(np.array(current_states[:,1].tolist()))
        # current_qs_list = self.prediction_predict(current_states)
        
        next_states = np.array([x[3] for x in self.stmemory])
        target_qs_list = self.target_predict(next_states)
        
        
        # next_qs_list = self.prediction_predict(next_states) #Double DQN
        results = self.prediction_predict(np.concatenate((current_states, next_states), axis=0))
        current_qs_list = results[0:len(current_states)]
        next_qs_list = results[len(current_states):]
        
        for i, (state, action, reward, next_state, done) in enumerate(self.stmemory):
            new_q = reward
            if not done:
                max_target_a = np.argmax(next_qs_list[i])
                # print(next_qs_list[i])
                max_target_q = target_qs_list[i][max_target_a] #Double DQN
                # max_target_q = np.amax(target_qs_list[i])
                new_q += self.discount_factor * max_target_q
                
            td_error = new_q - current_qs_list[i][action]
            
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
        
        info = self.replay()
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
      
    def replay(self):
        
        tic = time.perf_counter()
        loss = 0
        # if len(self.ltmemory) >= self.minibatch_size:
            # batch_size = min(len(self.ltmemory), self.minibatch_size)
            # batch = random.sample(self.ltmemory, batch_size)
            # batch = self.ltmemory
            
        for ii in range(1):
            # batch_size = min(len(self.stmemory), self.minibatch_size)
            # batch = random.sample(self.stmemory, batch_size)
            batch, idxs, is_weight = self.ltmemory.sample(self.minibatch_size)
            # print(batch, is_weight)
            
            current_states = np.array([x[0] for x in batch])
            # print(np.array(current_states[:,1].tolist()))
            # current_qs_list = self.prediction_predict(current_states)
            
            next_states = np.array([x[3] for x in batch])
            target_qs_list = self.target_predict(next_states)
            
            
            # next_qs_list = self.prediction_predict(next_states) #Double DQN
            results = self.prediction_predict(np.concatenate((current_states, next_states), axis=0))
            current_qs_list = results[0:len(current_states)]
            next_qs_list = results[len(current_states):]
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
                current_qs_list[i][action] = new_q
                
           
            # print(current_states[-1], y[-1])
            
            fit = self.model_prediction.fit(
                # fit_generator(current_states, current_qs_list, KERAS_BATCH_SIZE),
                [
                    np.array(current_states[:,0].tolist()), 
                    # np.array(current_states[:,1].tolist())
                ],
                current_qs_list,
                # sample_weight=is_weight,
                epochs=1,
                verbose=0,
                validation_split=0,
                # batch_size=batch_size,
                steps_per_epoch=None,
                # batch_size=32,
                workers = 32,
                use_multiprocessing=False
            )
            
            loss += fit.history['loss'][0]
        self.model_prediction.save_weights(self.weights_path)
        
        # Update target model counter every episode
        self.target_update_counter += 1
        # self.ltmemory.clear()
        
        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter >= self.update_target_every:
            print("Target is updated.")
            self.model_target.set_weights(self.model_prediction.get_weights())
            self.target_update_counter = 0
                
        toc = time.perf_counter()
        return {
            'loss': loss,
            'duration': toc - tic
        }
            
        
    def save_weights(self):
        self.model_prediction.save_weights(self.weights_path)
        print("Saved Weights")