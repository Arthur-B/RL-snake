import numpy as np
import random   # Can we get rid of that? In numpy probably


class env:
    def __init__(self, xSize=10, ySize=10):

        env.xSize, env.ySize = xSize, ySize
        env.gameOver = 0
        env.score = 0

        # Initialize the map, map is 0 where empty
        env.mapState = np.zeros((env.xSize, env.ySize), dtype=np.int)

        # Initialize the snake position and length
        xStart = int(np.rint(env.xSize / 4))
        xEnd = int(np.rint(env.xSize * 3/4))
        yStart = int(np.rint(env.ySize / 2)) - 1    # Middle of screen

        env.snake = np.array([[x, yStart] for x in range(xStart, xEnd)])

        # 1: snake body | 2: snake head
        env.mapState[env.snake[:, 0],  env.snake[:, 1]] = 1
        env.mapState[env.snake[-1, 0], env.snake[-1, 1]] = 2

        # Init the fruit position
        [xRandom, yRandom] = random.choice(np.argwhere(env.mapState == 0))
        env.mapState[xRandom, yRandom] = 3  # Map is 3 where there is fruit

        # Limit number of step (no turning in circle)
        env.maxMove = env.xSize * env.ySize * 2
        env.availableMove = env.maxMove

    def reset(self):
        """
        Put back into same state as initialized.

        """

        env.gameOver = 0
        env.score = 0

        # Init the map
        env.mapState = np.zeros((env.xSize, env.ySize), dtype=np.int)

        # Init the snake position and length
        xStart = int(np.rint(env.xSize / 4))
        xEnd = int(np.rint(env.xSize * 3/4))
        yStart = int(np.rint(env.ySize / 2)) - 1

        env.snake = np.array([[x, yStart] for x in range(xStart, xEnd)])

        # 1: snake body | 2: snake head
        env.mapState[env.snake[:, 0], env.snake[:, 1]] = 1
        env.mapState[env.snake[-1, 0], env.snake[-1, 1]] = 2

        # Init the fruit position
        [xRandom, yRandom] = random.choice(np.argwhere(env.mapState == 0))
        env.mapState[xRandom, yRandom] = 3  # Map is 3 where there is fruit

    def printState(self):
        """
        Print the map state: board, score, ...
        """

        print(env.mapState.T)
        print("Score: {}".format(env.score))

    def newFruit(self):
        """
        Place a new fruit on the map
        """

        if(np.argwhere(env.mapState == 0).size == 0):
            env.gameOver = 1
        else:
            [xRandom, yRandom] = random.choice(np.argwhere(env.mapState == 0))
            env.mapState[xRandom, yRandom] = 3

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
            xNew = env.snake[-1, 0] - 1
            yNew = env.snake[-1, 1]
        # Inverse up and down because we play in transpose
        elif keyPressed == 'w':
            xNew = env.snake[-1, 0]
            yNew = env.snake[-1, 1] - 1
        elif keyPressed == 'd':
            xNew = env.snake[-1, 0] + 1
            yNew = env.snake[-1, 1]
        elif keyPressed == 's':
            xNew = env.snake[-1, 0]
            yNew = env.snake[-1, 1] + 1

        # Put (xNew, yNew) within the grid boundaries
        xNew, yNew = xNew % env.xSize, yNew % env.ySize

        # If we move in the forbidden direction (on the neck of the snake)
        if [xNew, yNew] == [env.snake[-2, 0], env.snake[-2, 1]]:
            # We move forward
            xNew = env.snake[-1, 0] - (env.snake[-2, 0] - env.snake[-1, 0])
            yNew = env.snake[-1, 1] - (env.snake[-2, 1] - env.snake[-1, 1])
            # Re-apply the modulo
            xNew, yNew = xNew % env.xSize, yNew % env.ySize

        # Add new position to the snake
        env.snake = np.append(env.snake, [[xNew, yNew]], axis=0)

        # Update the map state

        if env.mapState[xNew, yNew] == 1:   # If stepping on another snake part
            env.reward = -1      # RL reward
            env.gameOver = 1
            print("Stepped on itself...")

        elif env.mapState[xNew, yNew] == 0:     # Normal case
            # snake moves, oldest point return to 0
            env.mapState[env.snake[0, 0], env.snake[0, 1]] = 0
            # Erase the oldest position
            env.snake = np.delete(env.snake, 0, axis=0)

            # Update new head on map
            env.mapState[env.snake[-1, 0], env.snake[-1, 1]] = 2
            # Previous head is now body
            env.mapState[env.snake[-2, 0], env.snake[-2, 1]] = 1

            # env.reward = -1   # Reward as in RL reward, empty move loose time
            env.reward = 0

        elif env.mapState[xNew, yNew] == 3:  # If snake reach a fruit
            # Do not erase oldest position
            # Update new point on map
            env.mapState[env.snake[-1, 0], env.snake[-1, 1]] = 2
            # Previous head is now body
            env.mapState[env.snake[-2, 0], env.snake[-2, 1]] = 1

            env.newFruit(self)      # Spawn a new fruit
            # env.reward = 100        # Maybe need to be tuned
            env.reward = 1  # Maybe need to be tuned
            env.score += 1  # Update game score, different from RL reward
            env.availableMove = env.maxMove  # The number of move goes back up

        # # Update the movement count
        # env.availableMove -= 1

        # if env.availableMove == 0:
        #     env.gameOver = 1
        #     print('No more moves...')

        return env.mapState, env.reward
