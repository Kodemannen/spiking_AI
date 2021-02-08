import numpy as np
import pygame as pg
from pygame.locals import *

class game_object:

    def __init__(self, position):
        pass




 
class CarGame:
    def __init__(self, win_size):
        pg.init()

        self._running = False
        #self.win_size = self.win_width, self.win_height = 800, 100
        self.win_size = self.win_width, self.win_height = win_size

        self.win = pg.display.set_mode(self.win_size)
        self.delay_ms = 10
 

        self.jump_vel = 5
        self.fall_vel = 5

        self.move_vel = 1
        self.jump_height = 45
        #self.jump_height = # roof is at 0
        #self.floor = self.win_height - self.player_height

        self.player_color = (153, 102, 255)


        #--------------------------------
        # Player params:
        #--------------------------------
        self.player_width = 20
        self.player_height = 10


        # vertical space between player/car and lane border
        space_vertical = (self.win_height/2 - self.player_height ) / 2 


        # Initializing player state
        self.player_pos_x = 20
        #self.player_pos_y = 65
        self.player_pos_y = self.win_height / 2 + space_vertical

        self.player_pos = (self.player_pos_x, self.player_pos_y)


        #self.jumping = False
        #self.falling = False

 
        #--------------------------------
        # Obstacle:
        #--------------------------------
        self.obstacle_size = (7, 30)              # width, height 
        self.obstacle_width, self.obstacle_height = self.obstacle_size
        self.obstacle_color = (153, 255, 187)
        self.obstacle_x = False
        self.obstacle_y = self.win_height - self.obstacle_height
        self.obstacle_vel = 5

        self.obstacle_spawn_prob = 0.03
        
        

    def update_player(self, keys):
        
        #-----------------------------------------
        # horizontal movement:
        #-----------------------------------------
        
        if keys[pg.K_h]==1:
            self.player_pos_x -= self.move_vel
        if keys[pg.K_l]==1:
            self.player_pos_x += self.move_vel


        #-----------------------------------------
        # vertical movement:
        #-----------------------------------------
        
        if keys[pg.K_k]==1:
            self.player_pos_y -= self.move_vel
        if keys[pg.K_j]==1:
            self.player_pos_y += self.move_vel

        return 0



    def update_obstacle(self):
        '''
        Update the position of the obstacle
        '''

        # Spawn or not:
        if self.obstacle_x == False: 
            dice = np.random.random()
            if dice <= self.obstacle_spawn_prob:
                # Start the right:
                self.obstacle_x = self.win_width
            else:
                pass

        # If already spawned:
        else:
            self.obstacle_x -= self.obstacle_vel

            # if exited screen:
            if self.obstacle_x < 0:
                self.obstacle_x = False

        return 0


    def evaluate(self):

        self.check_hit()



    def check_hit(self):
        
        #-------------------------------------------------
        # Check for hit
        #-------------------------------------------------

        obstacle_center_x = self.obstacle_x + self.obstacle_width/2
        player_center_x = self.player_pos_x + self.player_width/2
        print(self.obstacle_x)
        exit('hore')

        obstacle_center_y = self.obstacle_y + self.obstacle_height/2
        player_center_y = self.player_pos_y + self.player_height/2


        #-------------------------------------------------
        # check if too close in x dimension
        #-------------------------------------------------

        print(abs(player_center_x - obstacle_center_x))


        print(self.obstacle_width/2 + self.player_width/2)

        criterion1 = abs(player_center_x - obstacle_center_x) <= (self.obstacle_width/2 
                                                                 + self.player_width/2)


        


        exit('vada')


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
        '''
        Updating the positions of elements in the background
        '''
        pass



    def run_game_cycle(self, keys):

        self.update_player(keys)
        self.update_obstacle()
        self.update_background()
    
        return 0

    
    def render_player(self):
        pg.draw.rect(self.win, 
                     self.player_color, 
                     (self.player_pos_x, 
                      self.player_pos_y, 
                      self.player_width, 
                      self.player_height))
        return



    def render_obstacle(self):
        if not self.obstacle_x == False:
            pg.draw.rect(self.win, 
                         self.obstacle_color, 
                         (self.obstacle_x, 
                          self.obstacle_y, 
                          self.obstacle_width, 
                          self.obstacle_height))
        return


    def render_background(self):
        return


    def render(self):
        self.win.fill((0,0,0))   # clearing screen

        self.render_player()
        self.render_obstacle()
        self.render_background()

        pg.display.update()

        return


    def set_delay(self, delay):

        ''' 
        Set the value of the delay here in ms
        '''

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
        # check for key events like hits: 
        #---------------------------------------------
        self.evaluate()


        #---------------------------------------------
        # render graphics:
        #---------------------------------------------
        self.render()


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


    def draw_grid(self):

        #---------------------------------------------
        # - Draws a background grid that indicates the pixels 
        #     - Using pg.draw.lines()
        # - Assumes that self.grid exists
        #---------------------------------------------

        line_color = (153, 255, 187)    # RGB
        line_width = 5                  # 1 is default

        n_tot_lines = len(self.grid_lines)

        for i in range(n_tot_lines):

            print('wasd')
            line = self.grid_lines[i]

            start_line = line[0]       # (x, y) coordinate
            end_line   = line[1]       # (x, y) coordinate

            pg.draw.line(self.win, 
                         line_color, 
                         start_line, 
                         end_line, 
                         line_width)

        pass

 
if __name__ == "__main__" :
    #game = JumpGame()
    #game.play()
    pass





