# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 00:40:07 2021

*in progress*

@author: Arthur
"""

import torch
import numpy as np
import random


class env:
    def __init__(self, xSize=10, ySize=10):
        self.xSize, self.ySize = xSize, ySize
        self.reset()

    def reset(self):

        self.gameOver = 0
        self.score = 0

        # :,:,0: body, 1: head, 2: fruit
        self.mapState = torch.zeros((self.xSize, self.ySize, 3))

        # Define the starting snake position
        self.mapState[3, 4, 0] = 1
        self.mapState[4, 4, 0] = 1
        self.mapState[5, 4, 0] = 1
        self.mapState[5, 4, 1] = 1

        # Define the fruit position
        self.newFruit()
 
    def newFruit(self):
        """
        Place a new fruit on the map
        """

        if(np.argwhere(self.mapState[:, :, 0] == 0).size == 0):
            env.gameOver = 1  # If no empty space to put the fruit (no snake body), game over
        else:
            [xRandom, yRandom] = random.choice(np.argwhere(self.mapState[:, :, 0] == 0))
            self.mapState[xRandom, yRandom, 2] = 1

    def moveSnake(self, keyPressed):
        return self.mapState, self.reward