# -*- coding: utf-8 -*-
import gym
import numpy as np
from DQN import Agent

def try_gym():
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 10000
    agent = Agent(4, 2)
    observation = env.reset()
    while agent.begin_episode():
        done = False
        while not done:
            # env.render()
            state = observation.reshape(-1, 4)
            action = agent.get_action(state)
            observation, reward, done, _ = env.step(action)
            next_state = observation.reshape(-1, 4)
            agent.commit_memory(state, action, reward, done, next_state)
        observation = env.reset()
        agent.end_episode()
    env.close()
if __name__ == '__main__':
    try_gym()
