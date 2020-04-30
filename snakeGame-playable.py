import sys, pygame
from gameEnvironment import env # Import the game environment
import numpy as np

def drawGrid(mapState, offsetx, offsety, width, height, nbBlockx, nbBlocky, color, screen):
    
    blockSizex = width /nbBlockx #Set the size of the grid block
    blockSizey = height / nbBlocky
    
    # Draw grid
    for x in range(nbBlockx):
        for y in range(nbBlocky):
            rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                               blockSizex, blockSizey)
            pygame.draw.rect(screen, color, rect, 1) # Width of 1, empty rectangle
    
    # Draw snake body        
    for (x,y) in np.argwhere(gameEnv.mapState == 1):
        rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                           blockSizex, blockSizey)
        pygame.draw.rect(screen, (125,125,125), rect)
            
    # Draw snake head   
    for (x,y) in np.argwhere(gameEnv.mapState == 2):
        rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                           blockSizex, blockSizey)
        pygame.draw.rect(screen, (200,200,200), rect)
    
    # Draw fruit
    for (x,y) in np.argwhere(gameEnv.mapState == 3):
        rect = pygame.Rect(offsetx + x*blockSizex, offsety + y*blockSizey,
                           blockSizex, blockSizey)
        pygame.draw.rect(screen, (255,0,0), rect)

#==============================================================================
# Game
#==============================================================================

# Initialize the game environment

gameEnv = env(10,10)
gameEnv.printState()

# Initialize the game window

size = width, height = 600, 600
screen = pygame.display.set_mode(size)

colorBackground = 0, 0, 0
colorGrid       = 125, 125, 125

grid = pygame.Rect(50,50,500,500)

screen.fill(colorBackground)
drawGrid(gameEnv.mapState, 50, 50, 500, 500, 10, 10, (200, 200, 200), screen)
pygame.display.flip()

#------------------------------------------------------------------------------
# Game loop

while gameEnv.gameOver == 0:
    
    # Wait for a key press
    
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
    
    # Update the map state
                
    mapState, reward = gameEnv.moveSnake(keyPressed)
    
    # Update graphic
    
    screen.fill(colorBackground)
    drawGrid(gameEnv.mapState, 50, 50, 500, 500, 10, 10, (200, 200, 200), screen)
    pygame.display.flip()
    
#------------------------------------------------------------------------------
# Exit
    
pygame.time.delay(5000)
pygame.quit()
    



