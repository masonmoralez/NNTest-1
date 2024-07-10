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

while running:
    for event in p.event.get():
        if event.type == p.QUIT:
            running = False
        if p.mouse.get_pressed()[0]:
            # draw
            mouse_pos.append(p.mouse.get_pos())
            if len(mouse_pos) >= 2:
                mouse_pos[1] = mouse_pos[0]
                mouse_pos[0] = p.mouse.get_pos()
                p.draw.line(screen, WHITE, (mouse_pos[0]), mouse_pos[1], scale // 3)
            else:
                mouse_pos.append(p.mouse.get_pos())
        if not p.mouse.get_pressed()[0]:
                mouse_pos.clear()

        if event.type == p.KEYDOWN:
            # clear screen
            if event.key == p.K_c:
                screen.fill(BLACK)
                p.display.flip()
                mouse_pos.clear()
    p.display.flip()





p.quit()