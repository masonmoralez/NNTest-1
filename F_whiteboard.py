import pygame as py 

class WhiteBoard():
    def main():
        width = height = 784
        py.init()
        screen = py.display.set_mode((width, height))
        py.display.set_caption("AnimalChess")
        screen.fill(py.Color(117,117,117))