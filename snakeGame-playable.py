import sys, pygame
from gameEnvironment import env # Import the game environment
import numpy as np

#------------------------------------------------------------------------------
# Draw the map state

def drawGrid(mapState, offset, blockSize, nbBlockx, nbBlocky, screen):
    
    """
    Draw the mapState given the specifications.
    """
    # blockSize = 50
    # offset = 100
    
    # Draw grid
    for x in range(nbBlockx):
        for y in range(nbBlocky):
            rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                               blockSize, blockSize)
            pygame.draw.rect(screen, (140,140,140), rect, 1) # Width of 1, empty rectangle
    
    # Draw snake body        
    for (x,y) in np.argwhere(gameEnv.mapState == 1):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (222,132,82), rect)
            
    # Draw snake head   
    for (x,y) in np.argwhere(gameEnv.mapState == 2):
        rect = pygame.Rect(offset + x*blockSize, offset + y*blockSize,
                           blockSize, blockSize)
        pygame.draw.rect(screen, (197,78,82), rect)
    
    # Draw fruit
    for (x,y) in np.argwhere(gameEnv.mapState == 3):
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
    keyPressed=""

    while keyPressed=="":
        event = pygame.event.wait()
         
        if event.type == pygame.QUIT:
            pygame.display.quit()
            pygame.quit()
            sys.exit()
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

sizeX, sizeY = 10,5
blockSize = 50
offset = 50

size = (offset * 2 + sizeX * blockSize, offset * 2 + sizeY * blockSize)

# Initialize the game environment

gameEnv = env(sizeX,sizeY)
gameEnv.printState()

# Initialize the game window

screen = pygame.display.set_mode(size)
screen.fill((255,255,255))

drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen)

pygame.display.flip()

#------------------------------------------------------------------------------
# Game loop

while gameEnv.gameOver == 0:
    
    # Wait for a key press   
    keyPressed = getKeyPressed()
    
    # Update the map state               
    mapState, reward = gameEnv.moveSnake(keyPressed)
    
    # Update game window    
    screen.fill((255,255,255))
    drawGrid(gameEnv.mapState, offset, blockSize, sizeX, sizeY, screen)
    pygame.display.flip()
    
#------------------------------------------------------------------------------
# Exit
    
pygame.time.delay(5000)
pygame.quit()
    



