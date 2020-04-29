# import the pygame module, so you can use it
# import pygame
# 
import numpy as np
import pynput   # Keystroke
import random   # Can we get rid of that?


#------------------------------------------------------------------------------
# Detect keyPressed and return it as a global variable

def press(key):
    global keyPressed
    keyPressed = key.char
    return False

class env:
    def __init__(self):

#==============================================================================
# Game
#==============================================================================

#------------------------------------------------------------------------------
# Initialize parameters

# Game parameters

gameOver = 0
score = 0

# Map size

xSize = 10
ySize = 10

# Snake, array of [x,y] coordinates

snake = np.array([[2,5], [3,5], [4,5], [5,5]], dtype=np.int)
# snake[0, :] is the tail
# snake[-1, :] is the head
# snake[:, 0] is x-coordinate
# snake[:, 1] is y-coordinate

# Initialize the map, 0 if empty, 1 if there is snake

mapState = np.zeros((xSize, ySize), dtype=np.int)
mapState[snake[:,0], snake[:,1]] = 1

# Initialize a fruit on a random location

[xRandom, yRandom] = random.choice(np.argwhere(mapState==0))
mapState[xRandom, yRandom] = 2

# Show initial map

print(mapState.T, '\n')


#------------------------------------------------------------------------------
# Game loop

while gameOver == 0:

    # Movement

    with pynput.keyboard.Listener(on_press=press) as listener:
        listener.join()
    # Will return keyPressed 

    # if keyPressed == 'd':
    #     xNew = snake[-1, 0] + (snake[-2, 1] - snake[-1, 1]) # Use rotation matices
    #     yNew = snake[-1, 1] - (snake[-2, 0] - snake[-1, 0]) # flipped left and right because we play in mapState.T
    # if keyPressed == 'w':
    #     xNew = snake[-1, 0] - (snake[-2, 0] - snake[-1, 0])
    #     yNew = snake[-1, 1] - (snake[-2, 1] - snake[-1, 1])
    # if keyPressed == 'a':
    #     xNew = snake[-1, 0] - (snake[-2, 1] - snake[-1, 1])
    #     yNew = snake[-1, 1] + (snake[-2, 0] - snake[-1, 0])

    if keyPressed == 'a':
        xNew = snake[-1, 0] - 1
        yNew = snake[-1, 1]
    elif keyPressed == 'w': # Inverse up and down because we play in transpose
        xNew = snake[-1, 0]
        yNew = snake[-1, 1] - 1
    elif keyPressed == 'd':
        xNew = snake[-1, 0] + 1
        yNew = snake[-1, 1]
    elif keyPressed == 's':
        xNew = snake[-1, 0]
        yNew = snake[-1, 1] + 1

    # If we move in the forbidden direction (on the neck of the snake)
    if [xNew, yNew] == [snake[-2, 0], snake[-2, 1]]:
        xNew = snake[-1, 0] - (snake[-2, 0] - snake[-1, 0]) # We move forward
        yNew = snake[-1, 1] - (snake[-2, 1] - snake[-1, 1])

    # Check if out of bound

    xNew = np.mod(xNew, xSize)
    yNew = np.mod(yNew, xSize)

    # Add new position snake

    snake = np.append(snake, [[xNew, yNew]], axis=0)     # Add the new position

    if mapState[xNew, yNew] == 1: # If stepping on another snake part

        gameOver = 1
        print("Game Over\nScore:", score, "\n")

    elif mapState[xNew, yNew] == 0:             # Normal case

        mapState[snake[0,0], snake[0,1]] = 0    # snake moves, oldest point return to 0 
        snake = np.delete(snake, 0, axis=0)     # Erase the oldest position
        mapState[snake[-1,0], snake[-1,1]] = 1  # update new point on map

        print(mapState.T, '\n')

    elif mapState[xNew, yNew] == 2:
        # Do not erase oldest position
        mapState[snake[-1,0], snake[-1,1]] = 1  # update new point on map

        # Spawn a new fruit
        [xRandom, yRandom] = random.choice(np.argwhere(mapState==0))
        mapState[xRandom, yRandom] = 2

        # Update score
        score += 1

        print(mapState.T, '\n')



    



