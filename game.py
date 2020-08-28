import pygame as pg
from pygame.locals import *
 
class JumpGame:
    def __init__(self):
        pg.init()

        self._running = False
        self.win_size = self.win_width, self.win_height = 800, 150
        self.win = pg.display.set_mode(self.win_size)
        self.delay_ms = 10
 
        #--------------------------------
        # Player params:
        self.player_width = 20
        self.player_height = 35

        self.jump_vel = 5
        self.fall_vel = 5
        self.jump_height = 0            # roof is at 0
        self.floor = self.win_height - self.player_height

        self.player_color = (153, 102, 255)

        #--------------------------------
        # Initializing player state
        self.player_x = 50
        self.player_y = self.floor

        self.jumping = False
        self.falling = False

 

    def update_player(self, keys):
        if not (self.jumping or self.falling):
            if keys[pg.K_SPACE] == 1:
                self.player_y -= self.jump_vel
                self.jumping = True

        if self.jumping:
            self.player_y -= self.jump_vel
            if self.player_y <= self.jump_height:
                self.jumping = False
                self.falling = True

        if self.falling:
            self.player_y += self.fall_vel
            if self.player_y >= self.floor:
                self.player_y = self.floor
                self.falling = False

        return 0



    def update_obstacles(self):
        pass 

    def update_background(self):
        pass

    def run_game_cycle(self, keys):
        self.update_player(keys)
        self.update_obstacles()
        self.update_background()
    
        return 0

    
    def render_player(self):
        pg.draw.rect(self.win, 
                     self.player_color, 
                     (self.player_x, self.player_y, self.player_width, self.player_height))
    
        return 0


    def render_obstacles(self):
        pass

    def render_background(self):
        pass


    def render(self):
        self.win.fill((0,0,0))   # clearing screen

        self.render_player()
        self.render_obstacles()
        self.render_background()

        pg.display.update()
        return 0


    def delay(self):
        ''' 
                Induces a small delay between each timestep in the game
        '''
        pg.time.delay(self.delay_ms)
        return 0


    def quit_game(self):
        pg.quit()
        exit('exited game')

 
    def play(self):

        self._running = True 
        while( self._running ):

            self.delay()
            
            keys = pg.key.get_pressed()
            events = pg.event.get()

            self.run_game_cycle(keys)
            self.render()


            #--------------exit-----------------
            if keys[pg.K_q] == 1:
                self.quit_game()
            #-----------------------------------

        return 0

 
if __name__ == "__main__" :
    game = JumpGame()
    game.play()










































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
