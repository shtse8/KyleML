import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from keras.utils import Sequence
from keras.models import Sequential, Model
import keras.layers as layers
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

class Agent(object):
    def __init__(self, inputShape, outputShape, **kwargs):
        self.action_range = 1
        
        # Model
        self.inputShape = inputShape
        self.outputShape = outputShape
        
        # Trainning
        self.train = kwargs.get('train', True)
        self.episodes = 0
        self.target_episodes = kwargs.get('episodes', 150)
        self.episode_start_time = 0
        self.steps = 0
        self.total_rewards = 0
        self.highest_rewards = 0
        self.actor_weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + "_actor.h5")
        self.critic_weights_path = kwargs.get('weights_path', "./weights/" + os.path.basename(__main__.__file__) + "_critic.h5")
        self.update_target_every = kwargs.get('update_target_every', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.discount_factor = kwargs.get('discount_factor', 0.9)
        
        # Exploration
        self.epsilon_max = kwargs.get('epsilon_max', 1)
        self.epsilon_min = kwargs.get('epsilon_min', 0)
        self.epsilon_phase_size = kwargs.get('epsilon_phase_size', 0.8)
        self.epsilon = self.epsilon_max
        self.epsilon_decay = ((self.epsilon_max - self.epsilon_min) / (self.target_episodes * self.epsilon_phase_size))
        
        # Memory
        self.memory_size = kwargs.get('memory_size', 5000)
        
        # Mini Batch
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        
        print(f'epsilon_decay = {self.epsilon_decay}')
        self.train_overall_loss = []
        # self.ltmemory = collections.deque(maxlen=self.memory_size)
        self.ltmemory = Memory(self.memory_size)
        self.stmemory = collections.deque(maxlen=self.memory_size)
        
        # Prediction model (the main Model)
        self.critic = self.get_critic_model()
        
        self.critic_optimizer = Adam(0.002)
        self.actor_optimizer = Adam(0.001)

        print(self.critic.summary())
        self.actor = self.get_actor_model()
        print(self.actor.summary())
        
        # if os.path.exists(self.weights_path):
            # # Ask for clearing weights
            # clear_weights = input("Clear weights? (y/n)")
            # if clear_weights == "y":
                # try:
                    # os.remove(self.weights_path)
                # except:
                    # print("Failed to remove")
            # else:
                # try:
                    # self.critic.load_weights(self.critic_weights_path)
                    # self.actor.load_weights(self.actor_weights_path)
                    # print("weights loaded")
                # except:
                    # input("Failed to load weights. Continue?")
        
        # Target model
        self.critic_target = self.get_critic_model()
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor_target = self.get_actor_model()
        self.actor_target.set_weights(self.actor.get_weights())
        
        self.target_update_counter = 0
        
    # def optimizer(self, model, lr):
        # with tf.GradientTape() as tape:
            # action_gdts = K.placeholder(shape=(None, self.outputShape))
        # params_grad = tape.gradient(model.output, model.trainable_weights)
            # params_grad = tf.gradients(model.output, model.trainable_weights, -action_gdts)
        # grads = zip(params_grad, model.trainable_weights)
        # return K.function([model.input, action_gdts], [Adam(lr).apply_gradients(grads)])

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def get_action(self, state, greedy = False):
        a = round(self.actor_predict(np.array([state]))[0])
        if greedy:
            return a
        return np.clip(
            np.random.normal(a, self.epsilon), -self.action_range, self.action_range
        )  # add randomness to action selection for exploration

    
    
    def get_predict_action(self, state):
        prediction = self.prediction_predict(np.array([state]))
        # predict action based on the old state
        action = np.argmax(prediction[0])
        return action, prediction
    
    def get_actor_model(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(11,))
        out = layers.Dense(512, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation="relu")(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * 3
        model = tf.keras.Model(inputs, outputs)
        return model
        
        # state_input = Input(self.inputShape)
        # # x = Flatten()(x)
        # x = Dense(64, activation='relu')(state_input)
        # # x = Dropout(0.5)(x)
        # x = Dense(self.outputShape, activation="tanh")(x)
        # x = Dense(1)(x)
        
        # model = Model(inputs=[state_input], outputs=x)
        
        # # sys.exit(1)
        # return model
        
        inputs = Input(self.inputShape)
        out = Dense(64, activation="relu")(inputs)
        # out = layers.BatchNormalization()(out)
        # out = Dense(64, activation="relu")(out)
        # out = layers.BatchNormalization()(out)
        out = Dense(self.outputShape, activation="tanh")(out)
        out = Lambda(lambda x: 1 * x)(out)
        outputs = out
        # Our upper bound is 2.0 for Pendulum.
        # outputs = outputs * 3
        model = Model(inputs, outputs)
        return model
        
    def get_critic_model(self):
        # State as input
        state_input = layers.Input(shape=(11))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Dense(32, activation="relu")(state_out)
        state_out = layers.BatchNormalization()(state_out)

        # Action as input
        action_input = layers.Input(shape=(1))
        action_out = layers.Dense(32, activation="relu")(action_input)
        action_out = layers.BatchNormalization()(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(512, activation="relu")(concat)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(512, activation="relu")(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model
        
        # Define two input layers
        # print(self.inputShape, self.outputShape)
        state_input = Input(self.inputShape)
        action_input = Input(self.outputShape)
        x = Concatenate(1)([state_input, action_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(1)(x)
        # x = Dropout(0.5)(x)
        # x = Lambda(lambda x: action_range * x)(obseration_input)
        
        model = Model(inputs=[state_input, action_input], outputs=x)
        
        return model
        
    
    def actor_predict(self, states):
        # return self.model_prediction.predict(predict_generator(states, KERAS_BATCH_SIZE), steps=None, workers=32, use_multiprocessing=False)
        return self.actor([np.array(states[:,0].tolist())])

    def critic_predict(self, states, actions):
        # return self.model_target.predict(predict_generator(states, KERAS_BATCH_SIZE), steps=None, workers=32, use_multiprocessing=False)
        return self.critic.predict([
                np.array(states[:,0].tolist()), 
                np.array(actions[:,0].tolist())
            ], workers=32, use_multiprocessing=False)

    def actor_target_predict(self, states):
        # return self.model_prediction.predict(predict_generator(states, KERAS_BATCH_SIZE), steps=None, workers=32, use_multiprocessing=False)
        return self.actor_target.predict([
                np.array(states[:,0].tolist())
            ], workers=32, use_multiprocessing=False)

    def critic_target_predict(self, states, actions):
        # return self.model_target.predict(predict_generator(states, KERAS_BATCH_SIZE), steps=None, workers=32, use_multiprocessing=False)
        return self.critic_target.predict([
                np.array(states[:,0].tolist()), 
                np.array(actions[:,0].tolist())
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
            
            
            
        # current_states = np.array([x[0] for x in self.stmemory])
        # # print(np.array(current_states[:,1].tolist()))
        # # current_qs_list = self.prediction_predict(current_states)
        
        # next_states = np.array([x[3] for x in self.stmemory])
        # target_qs_list = self.target_predict(next_states)
        
        
        # # next_qs_list = self.prediction_predict(next_states) #Double DQN
        # results = self.prediction_predict(np.concatenate((current_states, next_states), axis=0))
        # current_qs_list = results[0:len(current_states)]
        # next_qs_list = results[len(current_states):]
        
        # for i, (state, action, reward, next_state, done) in enumerate(self.stmemory):
            # new_q = reward
            # if not done:
                # max_target_a = np.argmax(next_qs_list[i])
                # # print(next_qs_list[i])
                # max_target_q = target_qs_list[i][max_target_a] #Double DQN
                # # max_target_q = np.amax(target_qs_list[i])
                # new_q += self.discount_factor * max_target_q
                
            # td_error = new_q - current_qs_list[i][action]
            
            # # frame = collections.namedtuple('frame', ['state', 'action', 'reward', 'done', 'next_state'])
            # self.ltmemory.add(td_error, (
                # state,
                # action, 
                # reward, 
                # next_state,
                # done
            # )) 

            
        # self.stmemory.clear()
        
        info = self.replay()
        if self.total_rewards > self.highest_rewards:
            self.highest_rewards = self.total_rewards
            
        duration = time.perf_counter() - self.episode_start_time
        print(f'Episode {self.episodes:>5}/{self.target_episodes} | Loss: {info["loss"]:8.4f} | Rewards: {self.total_rewards:>5} (Max: {self.highest_rewards:>5}) | steps: {self.steps:>4} | Epsilon: {self.epsilon:>4.2f} | Time: {duration: >5.2f}')
        self.update_epsilon()
        return {
            'loss': info['loss'],
            'duration': duration
        }
      
    def replay(self):
        
        tic = time.perf_counter()
        loss = 0
        
        batch_size = min(len(self.stmemory), self.minibatch_size)
        batch = random.sample(self.stmemory, batch_size)
        
        # print(batch)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        with tf.GradientTape() as tape:
            next_actions = self.actor_target(next_states[:,0])
            next_q = self.critic_target([next_states[:,0], next_actions])
            # No need to care Done???
            y = rewards + self.discount_factor * next_q
            q = self.critic([states[:,0], actions])
            td_error = tf.losses.mean_squared_error(y, q)
            loss += sum(td_error.numpy())
        critic_grads = tape.gradient(td_error, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            a = self.actor(states[:,0])
            q = self.critic([states[:,0], a])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
            loss += actor_loss.numpy()
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update target model counter every episode
        self.target_update_counter += 1
        # self.ltmemory.clear()
        
        # If counter reaches set value, update target model with weights of main model
        if self.target_update_counter > self.update_target_every:
            # print("Target is updated.")
            self.actor_target.set_weights(self.actor.get_weights())
            self.critic_target.set_weights(self.critic.get_weights())
        self.target_update_counter = 0
                
        toc = time.perf_counter()
        return {
            'loss': loss,
            'duration': toc - tic
        }
            
        
    def save_weights(self):
        self.actor.save_weights(self.actor_weights_path)
        self.critic.save_weights(self.critic_weights_path)
        print("Saved Weights")