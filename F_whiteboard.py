import pygame as p

p.init()

scale = 20
screen_width = scale * 28
screen_height = scale * 28

'''
Colors
'''
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

screen = p.display.set_mode((screen_width, screen_height))
screen.fill(BLACK)
running = True
while running:
    for event in p.event.get():
        if event.type == p.QUIT:
            running = False
        if p.mouse.get_pressed()[0]:
            # draw
            mouse_x,mouse_y = p.mouse.get_pos()
            p.draw.circle(screen, WHITE, (p.mouse.get_pos()), scale // 3)
            print(p.mouse.get_pos())
        if event.type == p.KEYDOWN:
            # clear screen
            if event.key == p.K_c:
                screen.fill(BLACK)
                p.display.flip()
    p.display.flip()





p.quit()