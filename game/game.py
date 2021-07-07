import numpy as np
import pygame as pg
from pygame.locals import *

SPAWN_ISLAND = (999, 999)


class GameObject:
    #----------------------------------------
    # Class for a general object in the game
    #----------------------------------------

    def __init__(self, 
                pos=SPAWN_ISLAND, 
                vel=(2,1),
                size=(5,5),
                color=(153, 255, 187),
                ):

        self.pos_x, self.pos_y  = self.pos   = pos
        self.vel_x, self.vel_y  = self.vel   = vel

        self.width, self.height = self.size  = size
        self.color = color 


    def set_size(self, size):
        self.width, self.height = size


    def set_velocity(self, vel):
        self.vel_x, self.vel_y = vel





class Player(GameObject):
    pass


class Obstacle(GameObject):

    def __init__(self, 
                spawn_prob=0.3):

        super().__init__()
        self.spawn_prob = spawn_prob



 
class CarGame:
    def __init__(self, 
                win_size, 
                obstacle_size,
                n_lanes,
                n_cells_per_lane,
                background_lines=True):


        pg.init()


        self._running = False
        #self.win_size = self.win_width, self.win_height = 800, 100
        self.win_size = self.win_width, self.win_height = win_size

        #--------------------------------
        # main screen surface:
        #--------------------------------
        self.win = pg.display.set_mode(self.win_size)
        self.delay_ms = 10

        self.n_lanes = n_lanes
        self.n_cells_per_lane = n_cells_per_lane



        #--------------------------------
        # Player params:
        #--------------------------------

        #self.player   = GameObject()
        #self.obstacle = GameObject()

        #--------------------------------
        # Player 
        #--------------------------------
        self.player = Player(pos=(0,0), 
                            size=(20,10),
                            color=(153, 102, 255))



        #--------------------------------
        # Player params:
        #--------------------------------
        #self.player_width = 20
        #self.player_height = 10

        #self.move_vel_x = self.player_width
        #self.move_vel_y = self.player_height

        #self.player_color = (153, 102, 255)



        # vertical space between player/car and lane border
        space_vertical = (self.win_height/2 - self.player.height)/2 


        # Initializing player state
        #self.player_pos_x = 20
        ##self.player_pos_y = 65
        #self.player_pos_y = self.win_height / 2 + space_vertical

        #self.player_pos = (self.player_pos_x, self.player_pos_y)


        #self.jumping = False
        #self.falling = False

 
        #------------------------------------------------
        # Obstacle:
        #------------------------------------------------
        #obstacle_width, obstacle_height = obstacle_size
        self.obstacle = Obstacle(spawn_prob=0.005)
        self.obstacle.set_size(obstacle_size)

        #self.obstacle_size = (7, 40)              # width, height 




        #self.obstacle_width, self.obstacle_height = self.obstacle_size

        


        #--------------------------------
        # Background stuff
        #--------------------------------
        self.background_lines_on = False
        

    def update_player(self, keys):
        
        #-----------------------------------------
        # horizontal movement:
        #-----------------------------------------
        
        if keys[pg.K_h]==1:
            self.player.pos_x -= self.player.vel_x
        if keys[pg.K_l]==1:
            self.player.pos_x += self.player.vel_x


        #-----------------------------------------
        # vertical movement:
        #-----------------------------------------
        
        if keys[pg.K_k]==1:
            self.player.pos_y -= self.player.vel_y
        if keys[pg.K_j]==1:
            self.player.pos_y += self.player.vel_y

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


        if self.obstacle.pos_x == SPAWN_ISLAND[0]: 
            dice = np.random.random()

            if dice <= self.obstacle.spawn_prob:
                # Start the right:
                self.obstacle.pos_x = self.win_width

                lane = np.random.randint(1, self.n_lanes+1)
                # lane 1:
                self.obstacle.pos_y = (self.win_height/self.n_lanes)*lane - self.obstacle.height

                # lane 2:
                #self.obstacle.pos_y = self.win_height - self.obstacle_height

            else:
                pass


        #-----------------------------------------
        # if already spawned:
        #-----------------------------------------

        # Need a better way to make the obstacle 
        # only move on the grid 

        else:
            self.obstacle.pos_x -= self.obstacle.vel_x

            # if exited screen, return to spawn island:
            if self.obstacle.pos_x < -self.obstacle.width:
                self.obstacle.pos_x = SPAWN_ISLAND[0]

        return 0



    def evaluate(self):

        self.check_hit()
        pass



    def check_hit(self):
        
        #-------------------------------------------------
        # Check for hit
        #-------------------------------------------------

        obstacle_center_x = self.obstacle.pos_x + self.obstacle.width/2
        player_center_x = self.player.pos_x + self.player.width/2

        obstacle_center_y = self.obstacle.pos_y + self.obstacle.height/2
        player_center_y = self.player.pos_y + self.player.height/2


        #-------------------------------------------------
        # check if too close in x dimension
        #-------------------------------------------------

        criterion1 = abs(player_center_x - obstacle_center_x) <= (self.obstacle.width/2 
                                                                 + self.player.width/2)


        #-------------------------------------------------
        # check if too close in y dimension
        #-------------------------------------------------

        criterion2 = abs(player_center_y - obstacle_center_y) <= (self.obstacle.height/2 
                                                                 + self.player.height/2)


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

        self.playing = True

        self.update_player(keys)
        self.update_obstacle()
        self.update_background()
    
        return 


    
    def render_player(self):
        pg.draw.rect(self.win, 
                     self.player.color, 
                     (self.player.pos_x, 
                      self.player.pos_y, 
                      self.player.width, 
                      self.player.height))
        return



    def render_obstacle(self):
        if not self.obstacle.pos_x == 999:
            #self.obstacle.pos_x = 0

            #print((self.obstacle.pos_x, 
            #              self.obstacle.pos_y,
            #              self.obstacle.width, 
            #              self.obstacle.height))

            pg.draw.rect(self.win, self.obstacle.color, 
                         (self.obstacle.pos_x, 
                          self.obstacle.pos_y,
                          self.obstacle.width, 
                          self.obstacle.height))
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
        self.playing = False
        #exit('exited game')

 
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


    def add_background_lines(self):
        self.background_lines = self.create_grid_line_box()
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


    def create_grid_line_box(self):
        '''

        Creates a list containing the lines of a grid

        -----------------------------------------------------------------
        Input argument              : Type          | Description 
        -----------------------------------------------------------------

            n_lanes                 : integer       | number of car lanes
            n_cells_per_lane      : integer       | neurons per car lane

        '''

        n_cells_per_lane = self.n_cells_per_lane
        n_lanes = self.n_lanes
        win_size = self.win_size

        n_vertical_lines   = n_cells_per_lane + 1
        n_horizontal_lines = n_lanes + 1

        spacex = win_size[0]/n_cells_per_lane             # horizontal cell space 
        spacey = win_size[1]/n_lanes                        # vertical cell space

        startx = 0                                          # left vertical edge
        starty = 0                                          # top horizontal edge

        endx = startx + spacex*n_cells_per_lane           # right vertical edge
        endy = starty + spacey*n_lanes                      # bottom horizontal edge 

        line_box = []

        # Generating vertical lines:
        for i in range(n_vertical_lines):

            x = spacex*i + startx

            line = [(x, starty), (x, endy)]
            line_box.append(line)

        # Generating horizontal lines:
        for i in range(n_horizontal_lines):

            y = spacey*i + starty

            line = [(startx, y), (endx, y)]
            line_box.append(line)

        return np.array(line_box)


        
    def create_fov_lines(self, 
                         chosen_cells, 
                         spacex, 
                         spacey):

        #---------------------------------------------
        # Adds a different color for edges of the 
        # cells in the background grid that the snn
        # sees
        #---------------------------------------------

        # These could be useful:
        # self.win_width 
        # self.win_height
        # self.n_lanes 
        # self.n_cells_per_lane 

        # indexes count from top left to bottom right
        # i.e. there are n_

        fov_lines = []

        for ind in chosen_cells:

            # get the xy coordinate in the grid:
            x_index = ind % self.n_cells_per_lane 
            y_index = ind // self.n_lanes

            # create four lines for the cell:
            line1h = [[spacex*x_index, spacey*y_index], 
                      [spacex*(x_index+1), spacey*y_index]]

            line2h = [[spacex*x_index, spacey*(y_index+1)], 
                      [spacex*(x_index+1), spacey*(y_index+1)]]

            line1v = [[spacex*x_index, spacey*y_index], 
                      [spacex*x_index, spacey*(y_index+1)]]

            line2v = [[spacex*(x_index+1), spacey*y_index], 
                      [spacex*(x_index+1), spacey*(y_index+1)]]
        
            fov_lines.append([line1h, line2h, line1v, line2v])

        return fov_lines

 

if __name__ == "__main__" :
    #game = JumpGame()
    #game.play()
    pass





