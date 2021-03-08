# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:07:22 2021

Moving snake using tensor.
Adapted from:
    https://towardsdatascience.com/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1

Einstein summation in deep learning
    https://rockt.github.io/2018/04/30/einsum

@author: Arthur
"""

import torch
import torch.nn.functional as F


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

# Define 4 5x5 environment
heads = torch.zeros((4, 1, 5, 5))
# Place head in middle (2,2) coordinates
heads[:, 0, 2, 2] = 1

# 4 one hot vectors for direction
actions_onehot = torch.zeros((4, 4))
actions_onehot[torch.arange(4), torch.arange(4)] = 1

intermediate = F.conv2d(heads, movement_filters, padding=1)

# batch, channel, height, width
heads = torch.einsum('bchw,bc->bhw',
                     [intermediate, actions_onehot]).unsqueeze(1)
