import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class NumberRecognizer(nn.Module):
    def __init__(self):
        super(NumberRecognizer, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        # creates a grid that has 28 by 28 inputs (28 is the number of pixels)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        # 10 classes (digits 0 through 9)