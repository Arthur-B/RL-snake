import pygame
from gameEnvironment import env # Import the game environment
from numpy import argwhere

#------------------------------------------------------------------------------
# Draw the map state

def drawGrid(mapState, offset, blockSize, nbBlockx, nbBlocky, screen):
    
    """
    Draw the mapState given the specifications.
    """
        
    # Draw grid
    for x in range(nbBlockx):
        for y in range(nbBlocky):
            rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                               blockSize, blockSize)
            pygame.draw.rect(screen, (140,140,140), rect, 1) # Width of 1, empty rectangle
    
    # Draw snake body        
    for (x,y) in argwhere(gameEnv.mapState == 1):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (222,132,82), rect)
            
    # Draw snake head   
    for (x,y) in argwhere(gameEnv.mapState == 2):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (197,78,82), rect)
    
    # Draw fruit
    for (x,y) in argwhere(gameEnv.mapState == 3):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (76,114,176), rect)


#------------------------------------------------------------------------------
# Get key pressed

def getKeyPressed():
    
    """
    Wait until 'w', 'a', 's', 'd' is pressed and return the character as a string.
    """
    
    pygame.event.clear()
    keyPressed = ""

    while keyPressed=="":
        event = pygame.event.wait()
         
        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                keyPressed = 'w'
            elif event.key == pygame.K_a:
                keyPressed = 'a'
            elif event.key == pygame.K_s:
                keyPressed = 's'
            elif event.key == pygame.K_d:
                keyPressed = 'd'
    return keyPressed


#==============================================================================
# Game
#==============================================================================

#------------------------------------------------------------------------------
# Initialize variables

sizeX, sizeY = 10, 5    # Number of blocks width,height
blockSize = 50          # Pixel width/height of a unit block
offset = 50             # Pixel border

size = (offset * 2 + sizeX * blockSize, offset * 2 + sizeY * blockSize) # Overall size of screen 

# Initialize the game environment

gameEnv = env(sizeX,sizeY)  # Initialize the map
gameEnv.printState()        # Print underlying matrix of game

# Initialize the game window

screen = pygame.display.set_mode(size)  # Create game window of given size
screen.fill((255,255,255))              # Fill map

drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen)

pygame.display.flip()   # Update full display to screen

#------------------------------------------------------------------------------
# Game loop

while gameEnv.gameOver == 0:
    
    # Wait for a key press   
    keyPressed = getKeyPressed()
    
    # Update the map state               
    mapState, _ = gameEnv.moveSnake(keyPressed)
    gameEnv.printState()    # Print underlying matrix of game
    
    # Update game window    
    screen.fill((255,255,255))  # Reapply the background
    drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen) # Apply the latest grid
    pygame.display.flip()       # Update full display to screen
    
#------------------------------------------------------------------------------
# Exit
    
pygame.time.delay(5000)
pygame.quit()
    



