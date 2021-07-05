# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:01:12 2021

@author: Arthur Baucour
"""

import random
import math
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F

# =============================================================================
# DQN setup
# =============================================================================


class ReplayMemory(object):
    """
    """
    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.Transition = Transition
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(policy_net, n_actions, state, steps_done,
                  eps_start, eps_end, eps_decay):


    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

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


class SnakeNet(nn.Module):   # For 3x10x10 input

    def __init__(self):
        super(SnakeNet, self).__init__()

        self.feature_extraction = nn.Sequential(
                # 3x10x10
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3),
                nn.ReLU()
            )

        self.selection_fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)    # 3 outputs: left, forward, straight
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


def main():
    net = SnakeNet()

    test = torch.rand(1, 3, 10, 10)
    out = net(test)

if __name__ == '__main__':
    main()