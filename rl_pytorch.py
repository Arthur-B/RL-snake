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

# General DQN helper
from helper import ReplayMemory, select_action, SnakeNet

# Import the game environment
from game_environment.torch_no_parallel import env

from collections import namedtuple
from itertools import count
import torch
import torch.optim as optim
import torch.nn.functional as F


# =============================================================================
# Setup
# =============================================================================

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Game environment
sizeX, sizeY = 10, 10           # Number of blocks width,height
gameEnv = env(sizeX, sizeY)     # Initialize the map
# done = gameEnv.gameOver     # SHOULD BE CLEANED

# Replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
memory = ReplayMemory(10000, Transition)

# Network
policy_net = SnakeNet().to(device)
target_net = SnakeNet().to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Loss and optimizer
loss_fn = F.smooth_l1_loss
# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
# =============================================================================
# Training
# =============================================================================


# -----------------------------------------------------------------------------
# Hyperparameters

num_episodes = int(1e5)     # 50

batch_size = 128
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10

# n_actions = 4
n_actions = 3 # Left, forward, right




# -----------------------------------------------------------------------------
# Training loop

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

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

    # Get the state/action/reward batch
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
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = next_state_values.unsqueeze(1)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)   # Clip the gradient to -1, 1
    optimizer.step()


steps_done = 0
mean_score = []
mean_duration = []

for i_episode in range(num_episodes):
    # Initialize the environment and state
    gameEnv.reset()
    # adjust dtype?
    state = gameEnv.mapState

    for t in count():

        # Select and perform an action
        action = select_action(policy_net, n_actions, state, steps_done,
                               eps_start, eps_end, eps_decay)
        action = action.squeeze()

        # Get the reward (ditch the map state, stored in gameEnv.mapState)
        _, reward = gameEnv.moveSnake(action)
        reward = torch.tensor([reward], device=device) # Check the dimensions

        # Check if game is over
        done = gameEnv.gameOver
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
    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

        print("[{} / {}]\tMean duration: {:2.1f},\tMean score: {:3.1f}".format(
            i_episode, num_episodes, sum(mean_duration) / target_update,
            sum(mean_score) / target_update))

        mean_duration = []
        mean_score = []
        # plot_durations()


print('Complete')
# plot_durations()