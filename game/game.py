import numpy as np
import pygame as pg
from pygame.locals import *

class game_object:

    def __init__(self, position):
        pass




 
class JumpGame:
    def __init__(self):
        pg.init()

        self._running = False
        self.win_size = self.win_width, self.win_height = 150, 60
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
        self.obstacle_size = (7, 30)              # width, height 
        self.obstacle_width, self.obstacle_height = self.obstacle_size
        self.obstacle_color = (153, 255, 187)
        self.obstacle_x = False
        self.obstacle_y = self.win_height - self.obstacle_height
        self.obstacle_vel = 5

        self.obstacle_spawn_prob = 0.003
        
        

    def update_player(self, keys):
        
        #-----------------------------------------
        # Horizontal movement:
        if keys[pg.K_h]==1:
            self.player_pos_x -= self.move_vel
        if keys[pg.K_l]==1:
            self.player_pos_x += self.move_vel


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

        obstacle_center_x = self.obstacle_x + self.obstacle_width/2
        player_center_x = self.player_pos_x + self.player_width/2

        obstacle_center_y = self.obstacle_y + self.obstacle_height/2
        player_center_y = self.player_pos_y + self.player_height/2

        criterion1 = abs(player_center_x - obstacle_center_x) <= (self.obstacle_width/2 
                                                                 + self.player_width/2)

        criterion2 = abs(player_center_y - obstacle_center_y) <= (self.obstacle_height/2 
                                                                 + self.player_height/2)

        if criterion1 and criterion2:
            
            print('hit')
            print('game over')



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


    def delay(self):
        ''' 
                Induces a small delay between each timestep in the game
        '''
        pg.time.delay(self.delay_ms)
        return


    def quit_game(self):
        pg.quit()
        exit('exited game')

 
    def play_one_step(self):

        self.delay()
        
        keys = pg.key.get_pressed()
        events = pg.event.get()

        self.run_game_cycle(keys)
        self.evaluate()
        self.render()

        if keys[pg.K_q] == 1:
            #--------------exit-----------------
            self.quit_game()
            #-----------------------------------

        return 


    def get_pixels(self):
        #------------------------------------------------
        # Feed state to neural network
        surface = pg.display.get_surface()
        surface_array = pg.surfarray.array2d(surface)

        return surface_array


 
if __name__ == "__main__" :
    #game = JumpGame()
    #game.play()
    pass










































#
#
#import pygame as pg
#
# 
#class JumpGame:
#    def __init__(self):
#        pg.init()
#        self._running = True
#
#        #--------------------------------------------------
#        # initializing window
#        window_size = (400, 200)
#        self.win_width, self.win_height = window_size
#        self.win_size = window_size
#        self.win = pg.display.set_mode(window_size)
#
#
#        #--------------------------------------------------
#        # initializing the dude
#        self.dude_width = 20
#        self.dude_height = 35
#        self.x0 = 50
#        self.y0 = self.win_height - self.dude_height       # y is the top  
#
#        self.dude = dict(x=self.x0, 
#                         y=self.y0,
#                         jumping=False,
#                         falling=False,
#                         color = (153, 102, 255)
#                         ) 
#
#        #--------------------------------------------------
#        # other stuff
#        self.background = None
#        self.game_output = None
#
#        return None
#
#    def update_dude(self, keys):
#        #if keys[pg.K_q] == 
#        pass
# 
#    def draw_dude(self):
#
#        x = self.dude['x']
#        y = self.dude['y']
#        pg.draw.rect(self.win, color, (x, y, s))
#        return None
#        
#    def update_game(self, keys):
#        if keys[pg.K_q] == 1:
#            print('balllll')
#            #pg.quit()
#
#        return None
#
# 
#    def on_loop(self):
#        #win.
#        
#        pass
#
#    def on_render(self):
#        pass
#
#    def on_cleanup(self):
#        pg.quit()
# 
#    def play(self):
#        #if self.on_init() == False:
#        #    self._running = False
# 
#        while( self._running ):
#            keys = pg.key.get_pressed()
#            print(keys)
#            if keys[pg.K_q] == 1:
#                print('fitte')
#                print('quitting')
#                pg.quit()
#                exit('adasd')
#
#
#            self.update_game(keys)
#            #self.on_loop()
#            #self.on_render()
#
#            self.win.fill((0,0,0))
#            
#            pg.display.update()
#        self.on_cleanup()
# 
#if __name__ == "__main__" :
#    game = JumpGame()
#    game.play()
