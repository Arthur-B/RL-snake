import numpy as np
import random   # Can we get rid of that? In numpy probably


class env:
    def __init__(self, xSize=10, ySize=10):

        self.xSize, self.ySize = xSize, ySize
        self.reset()

        # Limit number of step (no turning in circle)
        # self.maxMove = self.xSize * self.ySize * 2
        # self.availableMove = self.maxMove

    def reset(self):
        """
        Put back into same state as initialized.

        """

        self.gameOver = 0
        self.score = 0

        # Init the map
        self.mapState = np.zeros((self.xSize, self.ySize), dtype=np.int)

        # Init the snake position and length
        xStart = int(np.rint(self.xSize / 4))
        xEnd = int(np.rint(self.xSize * 3/4))
        yStart = int(np.rint(self.ySize / 2)) - 1

        self.snake = np.array([[x, yStart] for x in range(xStart, xEnd)])

        # 1: snake body | 2: snake head
        self.mapState[self.snake[:, 0], self.snake[:, 1]] = 1
        self.mapState[self.snake[-1, 0], self.snake[-1, 1]] = 2

        # Init the fruit position
        self.newFruit()

    def printState(self):
        """
        Print the map state: board, score, ...
        """

        print(self.mapState.T)
        print("Score: {}".format(self.score))

    def newFruit(self):
        """
        Place a new fruit on the map
        """

        if(np.argwhere(self.mapState == 0).size == 0):
            env.gameOver = 1  # If no empty space to put the fruit, game over
        else:
            [xRandom, yRandom] = random.choice(np.argwhere(self.mapState == 0))
            self.mapState[xRandom, yRandom] = 3     # 3 is the fruit

    def moveSnake(self, keyPressed):
        """
        Move the snake on the map and return the current state of the game
        for the agent.

        Parameters
        ----------
        keyPressed : string ['a', 'w', 's', 'd']
            The input direction on where to move the snake

        Returns
        -------
        mapState: np.array
            The state of the game
        reward
            The reward of this specific move
        """

        if keyPressed == 'a':
            xNew = self.snake[-1, 0] - 1
            yNew = self.snake[-1, 1]
        # Inverse up and down because we play in transpose
        elif keyPressed == 'w':
            xNew = self.snake[-1, 0]
            yNew = self.snake[-1, 1] - 1
        elif keyPressed == 'd':
            xNew = self.snake[-1, 0] + 1
            yNew = self.snake[-1, 1]
        elif keyPressed == 's':
            xNew = self.snake[-1, 0]
            yNew = self.snake[-1, 1] + 1

        # Put (xNew, yNew) within the grid boundaries
        xNew, yNew = xNew % self.xSize, yNew % self.ySize

        # If we move in the forbidden direction (on the neck of the snake)
        if [xNew, yNew] == [self.snake[-2, 0], self.snake[-2, 1]]:
            # We move forward
            xNew = self.snake[-1, 0] - (self.snake[-2, 0] - self.snake[-1, 0])
            yNew = self.snake[-1, 1] - (self.snake[-2, 1] - self.snake[-1, 1])
            # Re-apply the modulo
            xNew, yNew = xNew % self.xSize, yNew % self.ySize

        # Add new position to the snake
        self.snake = np.append(self.snake, [[xNew, yNew]], axis=0)

        # Update the map state

        if self.mapState[xNew, yNew] == 1:  # If stepping on another snake part
            self.reward = -1      # RL reward
            self.gameOver = 1
            print("Stepped on itself...")

        elif self.mapState[xNew, yNew] == 0:     # Normal case
            # snake moves, oldest point return to 0
            self.mapState[self.snake[0, 0], self.snake[0, 1]] = 0
            # Erase the oldest position
            self.snake = np.delete(self.snake, 0, axis=0)

            # Update new head on map
            self.mapState[self.snake[-1, 0], self.snake[-1, 1]] = 2
            # Previous head is now body
            self.mapState[self.snake[-2, 0], self.snake[-2, 1]] = 1

            # env.reward = -1   # Reward as in RL reward, empty move loose time
            self.reward = 0

        elif self.mapState[xNew, yNew] == 3:  # If snake reach a fruit
            # Do not erase oldest position
            # Update new point on map
            self.mapState[self.snake[-1, 0], self.snake[-1, 1]] = 2
            # Previous head is now body
            self.mapState[self.snake[-2, 0], self.snake[-2, 1]] = 1

            self.newFruit()      # Spawn a new fruit
            # env.reward = 100        # Maybe need to be tuned
            self.reward = 1  # Maybe need to be tuned
            self.score += 1  # Update game score, different from RL reward

            # # The number of move goes back up
            # self.availableMove = self.maxMove

        # # Update the movement count
        # env.availableMove -= 1

        # if env.availableMove == 0:
        #     env.gameOver = 1
        #     print('No more moves...')

        return self.mapState, self.reward
