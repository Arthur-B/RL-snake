# import the pygame module, so you can use it
# import pygame
# 
# import numpy as np
import pynput   # Keystroke

from gameEnvironment import env # Import the game environment


# #------------------------------------------------------------------------------
# # Detect keyPressed and return it as a global variable

def press(key):
    global keyPressed
    keyPressed = key.char
    return False

#==============================================================================
# Game
#==============================================================================

# Initialize the environment

gameEnv = env(10,10)
gameEnv.printState()

# Game loop

while gameEnv.gameOver == 0:
    
    # Movement

    with pynput.keyboard.Listener(on_press=press) as listener:
        listener.join()
    # Will return keyPressed 
    # print(keyPressed)
    
    mapState, reward = gameEnv.moveSnake(keyPressed)
    


    



