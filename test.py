# allows us to do tensor math, calculations, and storage of data
import torch
# imports the neural networks classes and functions from pytorch
import torch.nn as nn

from train import training

# imports the optimization algorithms from pytorch like stochastic gradient descrent
# these are interchangeable with each other, just need to refer to each other
import torch.optim as optim
from torch.optim import SGD 
import torch.nn.functional as F 

# classes for handling datasets and creating data loaders
from torch.utils.data import Dataset, DataLoader

# allows you to normalize your data
import torchvision
import torchvision.transforms as transforms

# imports images
from torchvision.datasets import ImageFolder

# pre-trained image models from pytorch
# data visulization
import matplotlib.pyplot as plt

# data exploration
import pandas as pd

pixels = 28

# create neural network
# new class that inherits from nn.Module
class numNN_train(nn.Module):
    # new initialization method from the new class
    # creates and initializes the intial biases
    def __init__(self, in_features = (pixels ** 2), h1 = 16, h2 = 16, out_features = 10):
        # call initialization method for the parent class nn.Module
        super().__init__() # instantiate our nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # goes through the neural network by taking an input value and calculating the output value with the weights, biases, and activation functions
    def forward(self, x):
        # x = x.reshape(-1, 28 * 28) # Flattens the input (does this by setting the 2nd dimension to 756, and filling in the value for the first dimension)
        # this code moves everything forward thorugh the layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
# Need to pick a manual seed for randomization of the weights and biases that we are using
torch.manual_seed(11) # 11 is just a random number (in this case I picked it because its Jalen Brunson's number). Can really be any number
model = numNN_train()