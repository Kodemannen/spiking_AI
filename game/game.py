import numpy as np
import pygame as pg
from pygame.locals import *

class game_object:

    def __init__(self, position):
        pass




 
class CarGame:
    def __init__(self, win_size, 
                 obstacle_width, 
                 obstacle_height):

        pg.init()

        self._running = False
        #self.win_size = self.win_width, self.win_height = 800, 100
        self.win_size = self.win_width, self.win_height = win_size

        self.win = pg.display.set_mode(self.win_size)
        self.delay_ms = 10



        #--------------------------------
        # Player params:
        #--------------------------------
        self.player_width = 20
        self.player_height = 10

        self.move_vel_x = self.player_width
        self.move_vel_y = self.player_height

        self.player_color = (153, 102, 255)



        # vertical space between player/car and lane border
        space_vertical = (self.win_height/2 - self.player_height ) / 2 


        # Initializing player state
        self.player_pos_x = 20
        #self.player_pos_y = 65
        self.player_pos_y = self.win_height / 2 + space_vertical

        self.player_pos = (self.player_pos_x, self.player_pos_y)


        #self.jumping = False
        #self.falling = False

 
        #------------------------------------------------
        # Obstacle:
        #------------------------------------------------
        #self.obstacle_size = (7, 40)              # width, height 

        #------------------------------------------------
        # obstacle_size must be the grid size!
        #------------------------------------------------


        self.obstacle_size = obstacle_width, obstacle_height 
        self.obstacle_width, self.obstacle_height = self.obstacle_size

        self.obstacle_color = (153, 255, 187)

        self.obstacle_island = 999
        self.obstacle_x = self.obstacle_island    # just need some place far away
        self.obstacle_y = self.win_height - self.obstacle_height
        self.obstacle_vel = 5

        self.obstacle_spawn_prob = 0.3
        


        #--------------------------------
        # Background stuff
        #--------------------------------
        self.background_lines_on = False
        

    def update_player(self, keys):
        
        #-----------------------------------------
        # horizontal movement:
        #-----------------------------------------
        
        if keys[pg.K_h]==1:
            self.player_pos_x -= self.move_vel_x
        if keys[pg.K_l]==1:
            self.player_pos_x += self.move_vel_x


        #-----------------------------------------
        # vertical movement:
        #-----------------------------------------
        
        if keys[pg.K_k]==1:
            self.player_pos_y -= self.move_vel_y
        if keys[pg.K_j]==1:
            self.player_pos_y += self.move_vel_y

        return 0



    def update_obstacle(self):
        '''
        Update the position of the obstacle
        '''

        # Spawn or not:

        #-----------------------------------------
        # check if on island (storage place 
        # outside the game window 
        #-----------------------------------------

        if self.obstacle_x == self.obstacle_island: 
            dice = np.random.random()
            if dice <= self.obstacle_spawn_prob:
                # Start the right:
                self.obstacle_x = self.win_width
            else:
                pass


        #-----------------------------------------
        # if already spawned:
        #-----------------------------------------

        # Need a better way to make the obstacle 
        # only move on the grid 

        else:
            self.obstacle_x -= self.obstacle_vel

            # if exited screen:
            if self.obstacle_x < 0:
                self.obstacle_x = self.obstacle_island

        return 0


    def evaluate(self):

        self.check_hit()
        pass



    def check_hit(self):
        
        #-------------------------------------------------
        # Check for hit
        #-------------------------------------------------

        obstacle_center_x = self.obstacle_x + self.obstacle_width/2
        player_center_x = self.player_pos_x + self.player_width/2

        obstacle_center_y = self.obstacle_y + self.obstacle_height/2
        player_center_y = self.player_pos_y + self.player_height/2


        #-------------------------------------------------
        # check if too close in x dimension
        #-------------------------------------------------

        criterion1 = abs(player_center_x - obstacle_center_x) <= (self.obstacle_width/2 
                                                                 + self.player_width/2)


        #-------------------------------------------------
        # check if too close in y dimension
        #-------------------------------------------------

        criterion2 = abs(player_center_y - obstacle_center_y) <= (self.obstacle_height/2 
                                                                 + self.player_height/2)


        #-------------------------------------------------
        # both must be true for it to be a hit
        #-------------------------------------------------

        if criterion1 and criterion2:
            self.quit_game()

        return



    def update_background(self):
        
        #---------------------------------------------
        # Updating the positions of elements in the 
        # background
        #---------------------------------------------
        
        pass



    def run_game_cycle(self, keys):

        self.update_player(keys)
        self.update_obstacle()
        self.update_background()
    
        return 

    
    def render_player(self):
        pg.draw.rect(self.win, 
                     self.player_color, 
                     (self.player_pos_x, 
                      self.player_pos_y, 
                      self.player_width, 
                      self.player_height))
        return



    def render_obstacle(self):
        if not self.obstacle_x == 999:
            pg.draw.rect(self.win, 
                         self.obstacle_color, 
                         (self.obstacle_x, #- self.obstacle_width/2, 
                          self.obstacle_y, #- self.obstacle_height/2, 
                          self.obstacle_width, 
                          self.obstacle_height))
                          #1))
        return



    def render_background(self):
        if self.background_lines_on:
            self.draw_background_lines()
        return



    def render(self):
        self.win.fill((0,0,0))   # clearing screen

        self.render_player()
        self.render_obstacle()
        self.render_background()

        pg.display.update()

        return


    def set_delay(self, delay):

        #---------------------------------------------
        # Set the value of the delay here in ms
        #---------------------------------------------
       
        self.delay_ms = delay

        return


    def delay(self):
        ''' 
                Induces a small delay between each timestep in the game
        '''
        assert self.delay_ms, 'lacking delay value'

        pg.time.delay(self.delay_ms)
        return


    def quit_game(self):

        pg.quit()
        exit('exited game')

 
    def play_one_step(self):

        keys = pg.key.get_pressed()
        events = pg.event.get()


        #---------------------------------------------
        # run one simulation step:
        #---------------------------------------------
        self.run_game_cycle(keys)


        #---------------------------------------------
        # render graphics:
        #---------------------------------------------
        self.render()

        #---------------------------------------------
        # check for key events like hits: 
        #---------------------------------------------
        self.evaluate()


        #---------------------------------------------
        # exit by pressing q
        #---------------------------------------------
        if keys[pg.K_q] == 1:
            self.quit_game()


        #---------------------------------------------
        # delay
        #---------------------------------------------
        self.delay()

        return 




    def get_pixels(self):

        #--------------------------------------------
        # Feed state to neural network
        #--------------------------------------------

        surface = pg.display.get_surface()
        surface_array = pg.surfarray.array2d(surface)

        return surface_array


    def add_background_lines(self, background_lines):
        self.background_lines = background_lines
        self.background_lines_on = True

        return


    def draw_background_lines(self):



        #---------------------------------------------
        # - Draws a background grid that indicates the pixels 
        #     - Using pg.draw.lines()
        # - Assumes that self.grid exists
        #---------------------------------------------

        line_color = (153, 255, 187)    # RGB
        line_width = 1                  # 1 is default

        n_tot_lines = len(self.background_lines)

        for i in range(n_tot_lines):

            line = self.background_lines[i]

            start_line = line[0]        # (x, y) coordinate
            end_line   = line[1]        # (x, y) coordinate

            pg.draw.line(self.win, 
                         line_color, 
                         start_line, 
                         end_line, 
                         line_width)

        

 
if __name__ == "__main__" :
    #game = JumpGame()
    #game.play()
    pass





