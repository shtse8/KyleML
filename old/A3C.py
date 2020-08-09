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

import threading

tf.compat.v1.disable_eager_execution()

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(100, activation='relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(100, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense1(inputs)
    logits = self.policy_logits(x)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
  )
  result_queue.put(global_ep_reward)
  return global_ep_reward

class Agent():
  def __init__(self):
    self.game_name = 'CartPole-v0'
    save_dir = args.save_dir
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    env = gym.make(self.game_name)
    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n
    self.opt = tf.compat.v1.train.AdamOptimizer(args.lr, use_locking=True)
    print(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

  def train(self):
  
    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, game_name=self.game_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    plt.plot(moving_average_rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             '{} Moving Average.png'.format(self.game_name)))
    plt.show()

  def play(self):
    env = gym.make(self.game_name).unwrapped
    state = env.reset()
    model = self.global_model
    model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
    print('Loading model from: {}'.format(model_path))
    model.load_weights(model_path)
    done = False
    step_counter = 0
    reward_sum = 0

    try:
      while not done:
        env.render(mode='rgb_array')
        policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
        step_counter += 1
    except KeyboardInterrupt:
      print("Received Keyboard Interrupt. Shutting down.")
    finally:
      env.close()


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class Worker(threading.Thread):
  # Set up global variables across different threads
  global_episode = 0
  # Moving average reward
  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               game_name='CartPole-v0',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    self.game_name = game_name
    self.env = gym.make(self.game_name).unwrapped
    self.save_dir = save_dir
    self.ep_loss = 0.0

  def run(self):
    total_step = 1
    mem = Memory()
    while Worker.global_episode < args.max_eps:
      current_state = self.env.reset()
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      self.ep_loss = 0

      time_count = 0
      done = False
      while not done:
        logits, _ = self.local_model(
            tf.convert_to_tensor(current_state[None, :],
                                 dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        action = np.random.choice(self.action_size, p=probs.numpy()[0])
        new_state, reward, done, _ = self.env.step(action)
        if done:
          reward = -1
        ep_reward += reward
        mem.store(current_state, action, reward)

        if time_count == args.update_freq or done:
          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done,
                                           new_state,
                                           mem,
                                           args.gamma)
          self.ep_loss += total_loss
          # Calculate local gradients
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
          # Push local gradients to global model
          self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))
          # Update local model with new weights
          self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0

          if done:  # done and print information
            Worker.global_moving_average_reward = \
              record(Worker.global_episode, ep_reward, self.worker_idx,
                     Worker.global_moving_average_reward, self.result_queue,
                     self.ep_loss, ep_steps)
            # We must use a lock to save our model and to print to prevent data races.
            if ep_reward > Worker.best_score:
              with Worker.save_lock:
                print("Saving best model to {}, "
                      "episode score: {}".format(self.save_dir, ep_reward))
                self.global_model.save_weights(
                    os.path.join(self.save_dir,
                                 'model_{}.h5'.format(self.game_name))
                )
                Worker.best_score = ep_reward
            Worker.global_episode += 1
        ep_steps += 1

        time_count += 1
        current_state = new_state
        total_step += 1
    self.result_queue.put(None)

  def compute_loss(self,
                   done,
                   new_state,
                   memory,
                   gamma=0.99):
    if done:
      reward_sum = 0.  # terminal
    else:
      reward_sum = self.local_model(
          tf.convert_to_tensor(new_state[None, :],
                               dtype=tf.float32))[-1].numpy()[0]

    # Get discounted rewards
    discounted_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = self.local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))
    # Get our advantages
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                            dtype=tf.float32) - values
    # Value loss
    value_loss = advantage ** 2

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss

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
        self.actor_learning_rate = kwargs.get('actor_learning_rate', 0.001)
        self.critic_learning_rate = kwargs.get('critic_learning_rate', 0.01)
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
        
        
        
        # Build Models
        self.actor = self.get_actor_model()
        self.actor._make_predict_function()
        self.critic = self.get_critic_model()
        self.critic._make_predict_function()
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
    
    def get_actor_model(self):
        state_input = Input(self.inputShape, name="state_input")
        x = state_input
        x = Flatten()(state_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.outputShape, activation='softmax')(x)
        return Model(inputs=[state_input], outputs=output)
        
    def get_critic_model(self):
        state_input = Input(self.inputShape, name="state_input")
        x = state_input
        x = Flatten()(state_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        return Model(inputs=[state_input], outputs=output)
    
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