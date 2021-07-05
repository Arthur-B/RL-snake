# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:05:55 2020

To do:
    Clean plot, add score / time
    Move everyone to GPU tensor / scaling up
    Down to 8*8 grid, clean up network architecture
    Save / load, resume training of agent, etc.
    Move to solid walls?

@author: ArthurBaucour
"""

from helper import ReplayMemory, SnakeNet

# Import the game environment
from game_environment.numpy_no_parallel import env

import math
import random
# import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from collections import namedtuple
from itertools import count

import torch
# import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import torchvision. transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()   # Interactive mode
sns.set()
# If GPU with CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


# =============================================================================
# DQN setup
# =============================================================================


# -----------------------------------------------------------------------------
# Replay memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# -----------------------------------------------------------------------------
# Network


# =============================================================================
# Game environment
# =============================================================================


# -----------------------------------------------------------------------------
# Initialize variables

sizeX, sizeY = 10, 10    # Number of blocks width,height
blockSize = 1          # Pixel width/height of a unit block
offset = 0             # Pixel border

# Overall size of screen
size = (offset * 2 + sizeX * blockSize, offset * 2 + sizeY * blockSize)

# Initialize the game environment

gameEnv = env(sizeX, sizeY)  # Initialize the map
gameEnv.printState()        # Print underlying matrix of game

done = gameEnv.gameOver     # SHOULD BE CLEANED


# =============================================================================
# Training
# =============================================================================


# -----------------------------------------------------------------------------
# Hyperparameters and utilities

BATCH_SIZE = 32
# BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 200
TARGET_UPDATE = 20

n_actions = 4

# -----------------------------------------------------------------------------
# Define network

policy_net = SnakeNet().to(device)
target_net = SnakeNet().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100000, Transition)

steps_done = 0



episode_durations = []
episode_scores = []


def plot_durations():
    plt.figure(2)
    plt.clf()

    yPlot = pd.Series(episode_durations)
    yPlot_mean = yPlot.rolling(50).mean()

    plt.subplot(2, 1, 1)
    plt.plot(yPlot, '.')
    plt.plot(yPlot_mean, label='Rolling mean (50)')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.legend()

    yPlot = pd.Series(episode_scores)
    yPlot_mean = yPlot.rolling(50).mean()

    plt.subplot(2, 1, 2)
    plt.plot(yPlot, '.')
    plt.plot(yPlot_mean, label='Rolling mean (50)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()

    plt.pause(0.01)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# -----------------------------------------------------------------------------
# Training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                  batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either
    # the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 1000     # 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    gameEnv.reset()
    # adjust dtype?
    state = torch.tensor([[gameEnv.mapState]], dtype=torch.float32)

    for t in count():
        # Select and perform an action
        action = select_action(state)
        # Ugly key pressed
        if action[0] == 0:
            keyPressed = 'a'
        elif action[0] == 1:
            keyPressed = 'w'
        elif action[0] == 2:
            keyPressed = 's'
        elif action[0] == 3:
            keyPressed = 'd'
        else:
            print('wrong input:', action[0])

        _, reward = gameEnv.moveSnake(keyPressed)

        done = gameEnv.gameOver
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = torch.tensor([[gameEnv.mapState]], device=device,
                                      dtype=torch.float32)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            episode_scores.append(gameEnv.score)
            print("Episode:", i_episode, "Duration: ", t+1,
                  "Score: ", gameEnv.score, "\n")
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        plot_durations()


print('Complete')
plot_durations()
plt.ioff()
plt.show()