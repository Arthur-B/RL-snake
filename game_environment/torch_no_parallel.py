# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:40:07 2021

*in progress*

@author: Arthur
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

# Convolutions to move head in 4 directions
# 0, 1, 2, 3 = bottom, left, top, right
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

# (Body direction, action), with action in left, forward, right
dic_choose_tensor = {(0, 0): 3, (0, 1): 0, (0, 2): 1,
                     (1, 0): 0, (1, 1): 1, (1, 2): 2,
                     (2, 0): 1, (2, 1): 2, (2, 2): 3,
                     (3, 0): 2, (3, 1): 3, (3, 2): 0}

# 4 one hot vectors for direction
# actions_onehot = torch.zeros((4, 4))
# actions_onehot[torch.arange(4), torch.arange(4)] = 1


class GameEnvTorch:
    def __init__(self, xSize=5, ySize=5):
        self.xSize, self.ySize = xSize, ySize
        self.reset()

    def reset(self):

        self.game_over = 0
        self.score = 0

        # batch, channel (body, head, fruit), width, height
        # :,0,:,: body | :,1,:,: head | :,2,:,: fruit
        self.mapState = torch.zeros((1, 3, self.xSize, self.ySize))

        # Define the starting snake position
        self.mapState[0, 0, 1, 2] = 1   # Body
        self.mapState[0, 0, 2, 2] = 2
        self.mapState[0, 0, 3, 2] = 3
        self.mapState[0, 1, 3, 2] = 1   # Head

        # Body direction
        self.body_direction = 0    # Body starts toward bottom

        # Define the fruit position
        self.place_fruit()

    def place_fruit(self):
        """
        Place a new fruit on the map
        """

        # If no empty space to put the fruit (no snake body), game over
        if(np.argwhere(self.mapState[0, 0, :, :].numpy() == 0).size == 0):
            self.game_over = 1
        else:
            # Reinitialize the fruit map
            self.mapState[0, 2] = torch.zeros((self.xSize, self.ySize))
            # Find (x,y) where the body is 0 (= not on the snake)
            [xRandom, yRandom] = \
                random.choice(np.argwhere(self.mapState[0, 0, :, :].numpy() == 0))
            # Place a fruit in that random location
            self.mapState[0, 2, xRandom, yRandom] = 1

    def move_snake(self, keyPressed):
       
        keyPressed = keyPressed.item()

        action = dic_choose_tensor[self.body_direction, keyPressed]
        self.body_direction = action
        
        # Move the head
        self.mapState[0, 1] = \
            F.conv2d(self.mapState[0, 1].unsqueeze(0).unsqueeze(0),
                     movement_filters[action].unsqueeze(0),
                     padding=1).squeeze()

        # If the new head is on the fruit (growth)
        if ((self.mapState[0, 1] * self.mapState[0, 2]).max() == 1):
            # Place a new fruit
            self.place_fruit()
            # Add a body where is the updated head)
            self.mapState[0, 0].add_(self.mapState[0, 1]
                                     * (self.mapState[0, 0] + 1).max())

            # Write reward
            self.reward = 1
            self.score += 1

        # If the new head is on the body (game over)
        elif ((self.mapState[0, 0] * self.mapState[0, 1]).max() != 0):
            self.reward = -1
            self.game_over = 1
            # print('Stepped on itself')

        # If the head is out of bound (go through walls)
        elif (self.mapState[0, 1].max() == 0):
            self.reward = -1
            self.game_over = 1
            # print('Hit the wall')

        # Else, normal movement
        else:
            # Move the tail
            self.mapState[0, 0].sub_(1).relu_()
            # New front position (with updated head)
            self.mapState[0, 0].add_(self.mapState[0, 1]
                                     * (self.mapState[0, 0] + 1).max())
            self.reward = 0

        return self.mapState, self.reward

    def print_state(self):
        print((self.mapState[0, 0] - self.mapState[0, 2]).int())


# =========================================================================
# Check inputs
# =========================================================================

# def main():

#     sizeX, sizeY = 5, 5           # Number of blocks width,height
#     gameEnv = GameEnvTorch(sizeX, sizeY)
#     gameEnv.printState()

#     # 0, 1, 2, 3 = bottom, left, top, right
#     inputs = [0, 1, 2, 2]

#     for inp in inputs:
#         gameEnv.moveSnake(torch.tensor(inp))
#         # gameEnv.printState()
#         print("\n\nInput:\t", inp, "\n")
#         print(gameEnv.mapState)


# if __name__ == '__main__':
#     main()
