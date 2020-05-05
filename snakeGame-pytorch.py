# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:05:55 2020

@author: ArthurBaucour
"""

# import gym
import pygame
from gameEnvironment import env # Import the game environment
from numpy import argwhere

import math
import random
# import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd

from collections import namedtuple
from itertools import count
# from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision. transforms as T

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion() # Interactive mode

# If GPU with CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


#==============================================================================
# DQN setup
#==============================================================================


#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Network    
    
class Net(nn.Module): # For 3x10x10 input

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # 3x10x10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        # 6x6x6
        self.conv2 = nn.Conv2d(6, 16, 3)
        #16x4x4
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

 
class NetFC(nn.Module): # For 3x10x10 input

    def __init__(self):
        super(NetFC, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # 3x10x10

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3 * 10 * 10, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

#==============================================================================
# Game environment
#==============================================================================

#------------------------------------------------------------------------------
# Draw grid

def drawGrid(mapState, offset, blockSize, nbBlockx, nbBlocky, screen):
    
    """
    Draw the mapState given the specifications.
    """
        
    # Draw snake body        
    for (x,y) in argwhere(gameEnv.mapState == 1):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (222,132,82), rect)
            
    # Draw snake head   
    for (x,y) in argwhere(gameEnv.mapState == 2):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (197,78,82), rect)
    
    # Draw fruit
    for (x,y) in argwhere(gameEnv.mapState == 3):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (76,114,176), rect)

#------------------------------------------------------------------------------
# Initialize variables

sizeX, sizeY = 10, 10    # Number of blocks width,height
blockSize = 1          # Pixel width/height of a unit block
offset = 0             # Pixel border

size = (offset * 2 + sizeX * blockSize, offset * 2 + sizeY * blockSize) # Overall size of screen 

# Initialize the game environment

gameEnv = env(sizeX,sizeY)  # Initialize the map
gameEnv.printState()        # Print underlying matrix of game

done = gameEnv.gameOver     # SHOULD BE CLEANED

# Initialize the game window

screen = pygame.display.set_mode(size)  # Create game window of given size

screen.fill((255,255,255))              # Fill map
drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen)
pygame.display.flip()   # Update full display to screen
rawScreen = pygame.surfarray.array3d(screen)
refinedScreen = rawScreen.reshape(1, 3, sizeX, sizeY) / 255
state = torch.tensor(refinedScreen, dtype=torch.float32)

#==============================================================================
# Training
#==============================================================================
    
#------------------------------------------------------------------------------
## Hyperparameters and utilities

BATCH_SIZE = 128
# BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape
screen_height, screen_width = sizeX, sizeY

# Get number of actions from gym action space
# n_actions = env.action_space.n
n_actions = 4

# policy_net = DQN(screen_height, screen_width, n_actions).to(device)
# target_net = DQN(screen_height, screen_width, n_actions).to(device)
# policy_net = Net()
# target_net = Net()
policy_net = NetFC()
target_net = NetFC()

# # Test CUDA
# policy_net.cuda()
# target_net.cuda()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


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
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
episode_scores = []


def plot_durations():
    
    plt.figure(2)
    plt.clf()
    
    yPlot = pd.Series(episode_durations)
    yPlot_mean = yPlot.rolling(50).mean()
    
    plt.subplot(2,1,1)
    plt.plot(yPlot, '.')
    plt.plot(yPlot_mean, label='Rolling mean (50)')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.legend()
    
    yPlot = pd.Series(episode_scores)
    yPlot_mean = yPlot.rolling(50).mean()
    
    plt.subplot(2,1,2)
    plt.plot(yPlot, '.')
    plt.plot(yPlot_mean, label='Rolling mean (50)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()

    plt.pause(0.01)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
  
#------------------------------------------------------------------------------
## Training loop
        
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
                                          batch.next_state)), device=device, dtype=torch.bool)
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
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    

num_episodes = 1000 # 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    gameEnv.reset()
    screen.fill((255,255,255))              # Fill map
    drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen)
    pygame.display.flip()   # Update full display to screen
    rawScreen = pygame.surfarray.array3d(screen)
    refinedScreen = rawScreen.reshape(1, 3, sizeX, sizeY) / 255
    state = torch.tensor(refinedScreen, dtype=torch.float32)
    
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
            print('wrong input:', action[0] )
            
        # gameEnv.moveSnake(keyPressed)
        
        # _, reward, done, _ = gameEnv.step(action.item())
        _, reward = gameEnv.moveSnake(keyPressed)
        
        done = gameEnv.gameOver
        reward = torch.tensor([reward], device=device)

        # Observe new state
        # last_screen = current_screen
        # current_screen = get_screen()
        if not done:
            screen.fill((255,255,255))              # Fill map
            drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen)
            pygame.display.flip()   # Update full display to screen
            rawScreen = pygame.surfarray.array3d(screen)
            refinedScreen = rawScreen.reshape(1, 3, sizeX, sizeY) / 255
            next_state = torch.tensor(refinedScreen, dtype=torch.float32)
            # next_state = current_screen - last_screen
            # next_state = torch.tensor(gameEnv.mapState).cuda()
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
            print("Episode:", i_episode, "Duration: ",t+1, 
                  "Score: ", gameEnv.score, "\n")  
            break
      
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        plot_durations()


print('Complete')
pygame.time.delay(5000)
pygame.quit()
plot_durations()
plt.ioff()
plt.show()    
    