import pygame
import torch

from game_environment.torch_no_parallel import GameEnvTorch
from helper import make_gif, plot_game_state

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
                keyPressed = 1
            elif event.key == pygame.K_a:
                keyPressed = 0
            # elif event.key == pygame.K_s:
            #     keyPressed = 0
            elif event.key == pygame.K_d:
                keyPressed = 2
    return torch.tensor(keyPressed)


# =============================================================================
# Game
# =============================================================================

# -----------------------------------------------------------------------------
# Initialize variables

sizeX, sizeY = 5, 5  # Number of blocks width,height

# Initialize the game environment

gameEnv = GameEnvTorch(sizeX, sizeY)  # Initialize the map
gameEnv.print_state()  # Print underlying matrix of game

screen = pygame.display.set_mode((100, 100))  # necessary to get keypressed
# -----------------------------------------------------------------------------
# Game loop

i = 0
filenames = []
while gameEnv.game_over == 0:
    filename = f".\\images\\temp\\state_{i}.png"
    # Plot the map
    plot_game_state(gameEnv.mapState, filename)
    filenames.append(filename)
    i += 1

    # Update the game
    keyPressed = getKeyPressed()  # Wait for a key press
    mapState, _ = gameEnv.move_snake(keyPressed)  # Update the map state
    gameEnv.print_state()  # Print underlying matrix of game

# -----------------------------------------------------------------------------
# Exit

pygame.time.delay(5000)
pygame.quit()

make_gif(filenames, ".\\images\\played.gif")
