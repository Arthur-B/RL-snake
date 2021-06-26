# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 15:06:50 2021

To do:
    Clean plot, add score / time
    Move everyone to GPU tensor / scaling up
    Down to 8*8 grid, clean up network architecture
    Save / load, resume training of agent, etc.
    Move to solid walls?

@author: ArthurBaucour
"""

# Import the game environment
from game_environment.torch_no_parallel import env

import math
import random
# import numpy as np

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision. transforms as T

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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -----------------------------------------------------------------------------
# Network

class Net(nn.Module):   # For 3x10x10 input

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # 1x10x10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        # 6x6x6
        self.conv2 = nn.Conv2d(4, 8, 3)
        # 16x4x4
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(288, 120)  # 288 = 8*6*6 but why this value?
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.softmax(self.fc3(x), dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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

BATCH_SIZE = 128
# BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50

n_actions = 4

policy_net = Net()
target_net = Net()
# policy_net = Net().to(device)
# target_net = Net().to(device)

# # Test CUDA
# policy_net.cuda()
# target_net.cuda()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)



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
    # action_batch = torch.cat(batch.action)
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    # reward_batch = torch.cat(batch.reward)
    reward_batch = torch.tensor(batch.reward).unsqueeze(1)

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
    next_state_values = next_state_values.unsqueeze(1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 10000     # 50

mean_score = []
mean_duration = []

for i_episode in range(num_episodes):
    # Initialize the environment and state
    gameEnv.reset()
    # adjust dtype?
    state = gameEnv.mapState

    for t in count():

        # Select and perform an action
        action = select_action(state).squeeze()
        _, reward = gameEnv.moveSnake(action)

        done = gameEnv.gameOver
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = gameEnv.mapState
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            mean_score.append(gameEnv.score)
            mean_duration.append(t + 1)
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

        print("[{}]\tMean duration: {:2.1f},\tMean score: {:3.1f}".format(
            i_episode, sum(mean_duration) / TARGET_UPDATE,
            sum(mean_score) / TARGET_UPDATE))

        mean_duration = []
        mean_score = []
        # plot_durations()


print('Complete')
# plot_durations()