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

FRAME_RATE = 30

screen = p.display.set_mode((screen_width, screen_height))
screen.fill(BLACK)
running = True
mouse_pos = []
last_clicked = None
elapsed_time = 0
while running:
    for event in p.event.get():
        if event.type == p.QUIT:
            running = False
        if p.mouse.get_pressed()[0]:
            # draw
            if last_clicked is not None:
                elapsed_time = p.time.get_ticks() - last_clicked
            last_clicked = p.time.get_ticks()
            mouse_pos.append(p.mouse.get_pos())
            # makes sure line is continuous with 1.5x buffer
            if elapsed_time < FRAME_RATE * 1.5:
                if len(mouse_pos) >= 2:
                    mouse_pos[1] = mouse_pos[0]
                    mouse_pos[0] = p.mouse.get_pos()
                    p.draw.line(screen, WHITE, (mouse_pos[0]), mouse_pos[1], scale // 3)
                    print(p.mouse.get_pos())
                else:
                    mouse_pos.append(p.mouse.get_pos())
            else:
                mouse_pos.clear()

        if event.type == p.KEYDOWN:
            # clear screen
            if event.key == p.K_c:
                screen.fill(BLACK)
                p.display.flip()
                mouse_pos.clear()
    p.display.flip()





p.quit()