# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:15:28 2021

@author: Arthur
"""

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Definitions / constants

movement_filters = torch.Tensor([
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
]).unsqueeze(1).float()

# 4 one hot vectors for direction
actions_onehot = torch.zeros((4, 4))
actions_onehot[torch.arange(4), torch.arange(4)] = 1

# Snake: batch, head/body, xize, ysize
heads = torch.zeros((4, 1, 7, 7))
bodies = torch.zeros((4, 1, 7, 7))
num_envs = bodies.size(0)

#
heads[:, 0, 5, 5] = 1

bodies[:, 0, 3, 2] = 1
bodies[:, 0, 3, 3] = 2
bodies[:, 0, 2, 3] = 3
bodies[:, 0, 2, 4] = 4
bodies[:, 0, 2, 5] = 5
bodies[:, 0, 3, 5] = 6
bodies[:, 0, 4, 5] = 7
bodies[:, 0, 5, 5] = 8

print(bodies[1])

# -----------------------------------------------------------------------------
# Move the head

intermediate = F.conv2d(heads, movement_filters, padding=1)

# batch, channel, height, width
heads = torch.einsum('bchw,bc->bhw',
                     [intermediate, actions_onehot]).unsqueeze(1)

# -----------------------------------------------------------------------------
# Move the bodies

# Move the tail
bodies.sub_(1).relu_()

# New front position (with updated head)
bodies.add_(
    heads*(bodies+1).view(num_envs, -1).max(dim=1, keepdim=True)[0]
    .unsqueeze(-1)  # Add singleton dimensions so broadcasting works in add_
    .unsqueeze(-1)
)

print(bodies[1])
