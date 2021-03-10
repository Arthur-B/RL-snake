from game_environment.torch_no_parallel import env
import pygame
# import torch
# from numpy import argwhere


# -----------------------------------------------------------------------------
# Get key pressed

def getKeyPressed():
    """
    Wait until 'w', 'a', 's', 'd' is pressed and return the character
    as a string.
    """

    pygame.event.clear()
    keyPressed = ""

    while keyPressed == "":
        event = pygame.event.wait()

        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                keyPressed = 2
            elif event.key == pygame.K_a:
                keyPressed = 1
            elif event.key == pygame.K_s:
                keyPressed = 0
            elif event.key == pygame.K_d:
                keyPressed = 3
    return keyPressed


# =============================================================================
# Game
# =============================================================================

# -----------------------------------------------------------------------------
# Initialize variables

sizeX, sizeY = 10, 10    # Number of blocks width,height

# Initialize the game environment

gameEnv = env(sizeX, sizeY)  # Initialize the map
gameEnv.printState()        # Print underlying matrix of game

screen = pygame.display.set_mode((100, 100))  # necessary to get keypressed
# -----------------------------------------------------------------------------
# Game loop

while gameEnv.gameOver == 0:

    keyPressed = getKeyPressed()    # Wait for a key press
    mapState, _ = gameEnv.moveSnake(keyPressed)     # Update the map state
    gameEnv.printState()   # Print underlying matrix of game


# -----------------------------------------------------------------------------
# Exit

pygame.time.delay(5000)
pygame.quit()
