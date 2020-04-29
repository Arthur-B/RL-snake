# import the pygame module, so you can use it
# import pygame
# 
import numpy as np
import pynput   # Keystroke
import random   # Can we get rid of that? In numpy probably


#------------------------------------------------------------------------------
# Detect keyPressed and return it as a global variable

def press(key):
    global keyPressed
    keyPressed = key.char
    return False

class env:
    def __init__(self, xSize=10, ySize=10):
        
        env.xSize, env.ySize = xSize, ySize
        env.gameOver = 0
        env.score = 0
        
        # Init the map
        env.mapState = np.zeros((xSize, ySize), dtype=np.int)
        
        # Init the snake position
        env.snake = np.array([[2,5], [3,5], [4,5], [5,5]], dtype=np.int)
        env.mapState[env.snake[:,0], env.snake[:,1]] = 1
        
        # Init the fruit position
        [xRandom, yRandom] = random.choice(np.argwhere(env.mapState==0))
        env.mapState[xRandom, yRandom] = 2
        
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
        
        # Need a solution for when the map is full (win)
        
        [xRandom, yRandom] = random.choice(np.argwhere(env.mapState==0))
        env.mapState[xRandom, yRandom] = 2
        
    def moveSnake(self, keyPressed):
        """
        Move the snake on the map and return the current state of the game for the agent.

        Parameters
        ----------
        keyPressed : string ['a', 'w', 's', 'd']
            The input direction on where to move the snake

        Returns
        -------
        mapState: np.array
            The state of the game
        score
            Score accumulated until this point
        """
        
        if keyPressed == 'a':
            xNew = env.snake[-1, 0] - 1
            yNew = env.snake[-1, 1]
        elif keyPressed == 'w': # Inverse up and down because we play in transpose
            xNew = env.snake[-1, 0]
            yNew = env.snake[-1, 1] - 1
        elif keyPressed == 'd':
            xNew = env.snake[-1, 0] + 1
            yNew = env.snake[-1, 1]
        elif keyPressed == 's':
            xNew = env.snake[-1, 0]
            yNew = env.snake[-1, 1] + 1
    
        # If we move in the forbidden direction (on the neck of the snake)
        if [xNew, yNew] == [env.snake[-2, 0], env.snake[-2, 1]]:
            xNew = env.snake[-1, 0] - (env.snake[-2, 0] - env.snake[-1, 0]) # We move forward
            yNew = env.snake[-1, 1] - (env.snake[-2, 1] - env.snake[-1, 1])

        # Check if out of bound
    
        xNew = np.mod(xNew, env.xSize)
        yNew = np.mod(yNew, env.ySize)
    
        # Add new position snake
    
        env.snake = np.append(env.snake, [[xNew, yNew]], axis=0)     # Add the new position

        # Update the map state
        
        
        if env.mapState[xNew, yNew] == 1: # If stepping on another snake part
    
            env.gameOver = 1
            print("\nGame Over\nScore: {}".format(env.score))
    
        elif env.mapState[xNew, yNew] == 0:             # Normal case
    
            env.mapState[env.snake[0,0], env.snake[0,1]] = 0    # snake moves, oldest point return to 0 
            env.snake = np.delete(env.snake, 0, axis=0)     # Erase the oldest position
            env.mapState[env.snake[-1,0], env.snake[-1,1]] = 1  # update new point on map
    
            env.printState(self)
    
        elif env.mapState[xNew, yNew] == 2:  # If snake reach a fruit
            # Do not erase oldest position
            env.mapState[env.snake[-1,0], env.snake[-1,1]] = 1  # update new point on map
       
            env.newFruit(self)      # Spawn a new fruit           
            env.score += 1      # Update score
            env.printState(self)    # Print the map

        return env.mapState, env.score
#==============================================================================
# Game
#==============================================================================

# Initialize the environment

gameEnv = env()
gameEnv.printState()

# Game loop

while gameEnv.gameOver == 0:
    
    # Movement

    with pynput.keyboard.Listener(on_press=press) as listener:
        listener.join()
    # Will return keyPressed 
    # print(keyPressed)
    
    mapState, score = gameEnv.moveSnake(keyPressed)
    


    



