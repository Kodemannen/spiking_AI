import pygame as pg

 
class JumpGame:
    def __init__(self):
        pg.init()
        self._running = True

        #--------------------------------------------------
        # initializing window
        window_size = (400, 200)
        self.win_width, self.win_height = window_size
        self.win_size = window_size
        self.win = pg.display.set_mode(window_size)


        #--------------------------------------------------
        # initializing the dude
        self.dude_width = 20
        self.dude_height = 35
        self.x0 = 50
        self.y0 = self.win_height - self.dude_height       # y is the top  

        self.dude = dict(x=self.x0, 
                         y=self.y0,
                         jumping=False,
                         falling=False,
                         color = (153, 102, 255)
                         ) 

        #--------------------------------------------------
        # other stuff
        self.background = None
        self.game_output = None

        return None

    def update_dude(self, keys):
        #if keys[pg.K_q] == 
        pass
 
    def draw_dude(self):

        x = self.dude['x']
        y = self.dude['y']
        pg.draw.rect(self.win, color, (x, y, s))
        return None
        
    def update_game(self):
        if keys[pg.K_q] == 1:
            pg.quit()
        exit('hoooe')
        keys = pg.key.get_pressed()

        return None

 
    def on_loop(self):
        #win.
        
        pass

    def on_render(self):
        pass

    def on_cleanup(self):
        pg.quit()
 
    def play(self):
        #if self.on_init() == False:
        #    self._running = False
 
        while( self._running ):
            print('hore')
            self.update_game()
            #self.on_loop()
            #self.on_render()

            self.win.fill((0,0,0))
            
            pg.display.update()
        #self.on_cleanup()
 
if __name__ == "__main__" :
    game = JumpGame()
    game.play()
