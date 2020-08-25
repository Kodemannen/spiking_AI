import pygame as pg
pg.init()

win_width = 400
win_height = 200
win = pg.display.set_mode((400, 200))     # width, height
pg.display.set_caption('Selling lobs')

protagonist_width = 20
protagonist_height = 35
vel = 10

x0 = 50
y0 = win_height - protagonist_height

jump_vel = 20
fall_vel = 20

jumping = False
falling = False

ceiling = 70
floor = y0

delay_ms = 


c


x, y = x0, y0
run = True
while run:
    pg.time.delay(50)

    for event in pg.event.get():
        
        #if event.type == pg.QUIT: 
        #    run = False

        keys = pg.key.get_pressed()

        if keys[pg.K_q] == 1:
            pg.quit()


        if not (jumping or falling):
            if keys[pg.K_SPACE] == 1:
                y -= jump_vel
                jumping = True

    if jumping:
        y -= jump_vel
        if y <= ceiling:
            jumping = False
            falling = True

    if falling:
        y += fall_vel
        if y >= floor:
            y = floor
            falling = False


        #if keys[pg.K_UP] == 1 or keys[pg.K_k] == 1 :
        #    y -= vel

        #if keys[pg.K_DOWN] == 1 or keys[pg.K_j] == 1 :
        #    y += vel

    win.fill((0,0,0))
    pg.draw.rect(win, (153, 102, 255), (x, y, protagonist_width, protagonist_height))
    pg.display.update()
