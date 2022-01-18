# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:01:12 2021

@author: Arthur Baucour
"""

import math
import random
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn

# =============================================================================
# DQN setup
# =============================================================================


class ReplayMemory(object):
    """
    """

    def __init__(self, capacity, Transition):
        self.Transition = Transition
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(policy_net, n_actions, state, steps_done,
                  eps_start, eps_end, eps_decay):
    """
    Select an action to perform following an epsilon greedy policy.
    Careful, the decay is one the steps_done, not the number of iterations
    """

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


# =============================================================================
# Neural network
# =============================================================================


class SnakeNet(nn.Module):   # For 3x5x5 input

    def __init__(self):
        super(SnakeNet, self).__init__()

        self.feature_extraction = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, 1),
                nn.ReLU()
            )

        self.selection_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)    # 3 outputs: left, forward, right
            )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = self.selection_fc(x)
        # x = F.softmax(x, dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# =============================================================================
# Plot
# =============================================================================

def make_plot(x_plot, duration_list, score_list):
    # sns.set()
    # Process the data
    df = pd.DataFrame({"Episode": x_plot, "Duration":duration_list, "Score": score_list})
    df_rolling = df.rolling(5).mean() # Get the rolling average, using 5 points
    
    # Build plot
    fig, ax1 = plt.subplots(figsize=(5.4, 4), dpi=300)
    ax2 = ax1.twinx() # For xyy plot
    axs = [ax1, ax2]

    ax1.scatter(x_plot, duration_list, marker='.', color='C0')
    ax1.plot(df_rolling["Episode"], df_rolling["Duration"], color='C0')
    ax2.scatter(x_plot, score_list, marker='.', color='C1')
    ax2.plot(df_rolling["Episode"], df_rolling["Score"], color='C1')

    # Fluff xyy plot
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration', color='C0')
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')

    ax2.set_ylabel('Score', color='C1')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')
    ax2.spines['right'].set_color('C1')
    ax2.spines['left'].set_color('C0')

    fig.tight_layout()
    fig.savefig('training_process.png')
    
    return fig, axs

# =============================================================================
# Main
# =============================================================================


# def main():
#     net = SnakeNet()

#     test = torch.rand(1, 3, 5, 5)
#     out = net(test)


# if __name__ == '__main__':
#     main()

