import pygame as pg
from pygame.locals import *
 
class App:
    def __init__(self):
        pg.init()

        self._running = True
        self.win_size = self.win_width, self.win_height = 640, 200

        self.delay_ms = 10
 
        self.win = pg.display.set_mode(self.win_size)
        self._running = True

        #--------------------------------
        # Player params:
        self.player_width = 20
        self.player_height = 35

        self.player_x = 50
        self.player_y = self.win_height - self.player_height

        self.jump_vel = 20
        self.fall_vel = 20

        self.jumping = False
        self.falling = False

        self.jump_height = 70
 

    def player(self):
        pass


    def on_event(self, event):
        if event.type == pg.QUIT:
            self._running = False


    def on_loop(self):
        pass
    

    def on_render(self):
        pass


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
        #if self.on_init() == False:
        #    self._running = False
 
        while( self._running ):
            self.delay()
            
            keys = pg.key.get_pressed()

            


            for event in pg.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()


            #--------------exit-----------------
            if keys[pg.K_q] == 1:
                self.quit_game()
            #-----------------------------------

        return 0

 
if __name__ == "__main__" :
    theApp = App()
    theApp.play()










































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
