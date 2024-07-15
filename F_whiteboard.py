import pygame as p
import torch
from _test import AdvancedNN
from PIL import Image

def guess(screen, network):
    scaled_screen = p.transform.scale(screen, (screen_width / scale, screen_height / scale))
    p.image.save(scaled_screen, "drawing.png")
    image = p.image.load("drawing.png")
    grayscale_array = []
    for y in range(28):
        for x in range(28):
            color = image.get_at((x, y))
            grayscale_array.append(int(rgb_to_grayscale(color.r,color.g,color.b)))
    
    # Convert the list to a PyTorch tensor and add batch dimension
    input_tensor = torch.tensor(grayscale_array).unsqueeze(0).float()
    
    # Make the prediction using the model
    with torch.no_grad():
        output = network(input_tensor)
    
    # Get the predicted class
    prediction = torch.argmax(output, dim=1).item()
    print("Prediction:", prediction)

    # Render the text
    text = FONT.render(str(prediction), True, WHITE)  # White text
    text_rect = text.get_rect(center=(screen_height * 0.05, screen_width * 0.05))
    screen.blit(text, text_rect)

def rgb_to_grayscale(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

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

FONT = p.font.Font(None, 40)

screen = p.display.set_mode((screen_width, screen_height))
screen.fill(BLACK)
running = True
mouse_pos = []

network = AdvancedNN()
network.load_state_dict(torch.load('optimized_NN.pth'))
network.eval()

while running:
    for event in p.event.get():
        if event.type == p.QUIT:
            running = False
        if p.mouse.get_pressed()[0]:
            p.draw.circle(screen, WHITE, (p.mouse.get_pos()), 8.5)
            # draw
            mouse_pos.append(p.mouse.get_pos())
            if len(mouse_pos) >= 2:
                mouse_pos[1] = mouse_pos[0]
                mouse_pos[0] = p.mouse.get_pos()
                p.draw.line(screen, WHITE, (mouse_pos[0]), mouse_pos[1], scale * 2)
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
            if event.key == p.K_g:
                guess(screen, network)
    p.display.flip()
p.quit()