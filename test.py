import os
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

# Custom dataset class
class CustomDigitDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        # Assuming the first column is labels and the rest are pixel values
        self.labels = torch.tensor(self.data_frame.iloc[:, 0].values, dtype=torch.long)
        print("Labels: ", self.labels)
        self.features = torch.tensor(self.data_frame.iloc[:, 1:].values, dtype=torch.float32)
        print("Features: ", self.features)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
train_dataset = CustomDigitDataset(os.path.join(base_path,'train.csv'))
test_dataset = CustomDigitDataset(os.path.join(base_path,'test.csv'))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    # - train_dataset (the dataset that we loaded)
    # - batch_size (how many samples will be passed through the data at one time)
    # - shuffle (enabling this with True means that you get random data each time)
print(train_loader)
print(test_loader)

''' Uncomment this code for debugging (purpose is to see the labels and images plus their size)'''
'''
# Print some samples from train_loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)
print('Labels:', labels)

# Optionally, you can print the images in a more readable format (for example, as numpy arrays)
print('Images:', images.numpy())'''

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

