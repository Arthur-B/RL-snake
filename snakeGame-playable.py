import sys, pygame
from gameEnvironment import env # Import the game environment
import numpy as np

#------------------------------------------------------------------------------
# Draw the map state

def drawGrid(mapState, offsetx, offsety, width, height, nbBlockx, nbBlocky, screen):
    
    """
    Draw the mapState given the specifications.
    """
    
    blockSizex = width /nbBlockx #Set the size of the grid block
    blockSizey = height / nbBlocky
    
    # Draw grid
    for x in range(nbBlockx):
        for y in range(nbBlocky):
            rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                               blockSizex, blockSizey)
            pygame.draw.rect(screen, (140,140,140), rect, 1) # Width of 1, empty rectangle
    
    # Draw snake body        
    for (x,y) in np.argwhere(gameEnv.mapState == 1):
        rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                           blockSizex, blockSizey)
        pygame.draw.rect(screen, (222,132,82), rect)
            
    # Draw snake head   
    for (x,y) in np.argwhere(gameEnv.mapState == 2):
        rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                           blockSizex, blockSizey)
        pygame.draw.rect(screen, (197,78,82), rect)
    
    # Draw fruit
    for (x,y) in np.argwhere(gameEnv.mapState == 3):
        rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                           blockSizex, blockSizey)
        pygame.draw.rect(screen, (76,114,176), rect)


#------------------------------------------------------------------------------
# Get key pressed

def getKeyPressed():
    
    """
    Wait until 'w', 'a', 's', 'd' is pressed and return the character.
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

# Initialize the game environment

gameEnv = env(10,10)
gameEnv.printState()

# Initialize the game window

size = width, height = 600, 600
screen = pygame.display.set_mode(size)

grid = pygame.Rect(50,50,500,500)

screen.fill((255,255,255))
drawGrid(gameEnv.mapState, 50, 50, 500, 500, 10, 10, screen)
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
    drawGrid(gameEnv.mapState, 50, 50, 500, 500, 10, 10, screen)
    pygame.display.flip()
    
#------------------------------------------------------------------------------
# Exit
    
pygame.time.delay(5000)
pygame.quit()
    



