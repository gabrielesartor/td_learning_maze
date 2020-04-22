import sys
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import pickle

import gym
import gym_maze

"""
Implementation of TD methods for the maze environment.
(you can find the environment here: https://github.com/MattChanTK/gym-maze)
"""

#Simulation parameters
NUM_EPISODES = 10000
MAX_T = 100
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.2

#Test flgas
DEBUG = False
RENDER_POLICY = True
MEAN_RANGE = 10
MIN_REWARD = -0.1
MAX_REWARD = 1.0
NUM_EPISODES_PLOT = 1000
ALGORITHM = "sarsa" # "q-learning", "sarsa" or "expected-sarsa"

def select_action_e_greedy(env, s, Q_values, epsilon = 0.5):
    """
    It selects a random action with probability epsilon, otherwise the best learned until now.
    """
    if np.random.rand() < epsilon:
        return env.ACTION[np.random.randint(len(env.ACTION))]
    else:
        return env.ACTION[np.argmax(Q_values[s[0],s[1],:])]

def learning_episode(env, Q_values, algorithm = "q-learning"):
    """
    It calculates a whole simulation episode using the algorithm passed as argument.
    """
    total_reward = 0

    for t in range(MAX_T):
        s = [env.maze_view.robot[0], env.maze_view.robot[1]]
        a = select_action_e_greedy(env, s, Q_values, epsilon = EPSILON)

        s_new, reward, done, _ = env.step(a)

        Q_prev = Q_values[s[0],s[1],env.ACTION.index(a)]

        #updata rule
        if algorithm == "q-learning":
            next_max_Q = np.max(Q_values[s_new[0],s_new[1],:])
            Q_values[s[0],s[1],env.ACTION.index(a)] = Q_prev + ALPHA*(reward + GAMMA*next_max_Q - Q_prev)
        elif algorithm == "sarsa":
            next_a = Q_values[s_new[0],s_new[1],np.random.randint(len(env.ACTION))]
            Q_values[s[0],s[1],env.ACTION.index(a)] = Q_prev + ALPHA*(reward + GAMMA*next_a - Q_prev)
        elif algorithm == "expected-sarsa":
            next_expected_value = np.mean(Q_values[s_new[0],s_new[1],:]) #it is not correct because the actions don't have the same p of being executed but it is quite reasonable
            Q_values[s[0],s[1],env.ACTION.index(a)] = Q_prev + ALPHA*(reward + GAMMA*next_expected_value - Q_prev)
        else:
            raise NameError('Unknown algorithm name!')

        if DEBUG:
            print("action: ", a)
            print("s_new: ", s_new)
            print("q_values", Q_values[s_new[0],s_new[1],:])
            print("max q: ", np.max(Q_values[s_new[0],s_new[1],:]))
            print("actions: ", env.ACTION)
            print("index a: ", env.ACTION.index(a))
            print("Q_prev: ", Q_prev)
            print("Q_succ: ", Q_values[s[0],s[1],env.ACTION.index(a)])
            print("reward :", reward)

        total_reward = total_reward + reward

        if done:
            break

    return total_reward

def training(env, Q_values, rewards, algorithm = "q-learning"):
    """
    It carries out the training phase executing NUM_EPISODES of trials in the environment.
    """
    for episode in range(NUM_EPISODES):
        env.reset()

        episode_reward = learning_episode(env, Q_values, algorithm)
        rewards[episode] = episode_reward

        if episode % NUM_EPISODES_PLOT == 0 and episode!=0:
            plt.plot(range(episode+1), [np.mean(rewards[max(0,i-MEAN_RANGE):min(i+MEAN_RANGE,episode+1)]) for i in range(episode+1)], "b")
            plt.axis([0, episode+1, MIN_REWARD, MAX_REWARD])
            plt.pause(0.05)

            if RENDER_POLICY:
                render_policy(env, Q_values)

def render_policy(env, Q_values, epsilon = 0):
    """
    It shows the current learned behaviour on the GUI
    """
    env.reset()

    for t in range(MAX_T):
        env.render()

        s = [env.maze_view.robot[0], env.maze_view.robot[1]]
        a = select_action_e_greedy(env, s, Q_values, epsilon)
        s_new, reward, done, _ = env.step(a)

        time.sleep(0.1)

        if done:
            print("I've reached the goal!")
            break

    print("Policy executed.")
    env.render()

#COPY THE ENVIRONMENT YOU PREFER!
# env = gym.make("maze-random-10x10-v0")
# env = gym.make("maze-random-20x20-plus-v0")
# env = gym.make("maze-random-30x30-plus-v0")
# env = gym.make("maze-random-100x100-v0")

if __name__ == "__main__":
    # Initialize the "maze" environment
    env = gym.make("maze-random-20x20-plus-v0")

    env_dim = [env.maze_size[0],env.maze_size[1]]
    rewards = np.zeros(NUM_EPISODES)
    Q_values = np.zeros((env_dim[0], env_dim[1], len(env.ACTION)))

    training(env, Q_values, rewards, algorithm = ALGORITHM)

    print("Execute final policy...")
    render_policy(env, Q_values)
    print("Everything is done!")

    plt.show()
