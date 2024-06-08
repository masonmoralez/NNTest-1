import os

# allows us to do tensor math, calculations, and storage of data
import torch

# imports the neural networks classes and functions from pytorch
import torch.nn as nn

# imports the optimization algorithms from pytorch like stochastic gradient descrent
# these are interchangeable with each other, just need to refer to each other
import torch.optim as optim
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
import numpy as np

import csv


# Step 2: Define your neural network architecture
class numberDataset (Dataset):
    # initializes the data setting it to the csv_file that is being read
    def __init__(self, csv_file, transform=None):
        # reads in data
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # returns

    def __len__(self):
        return len(self.data)
        # returns the length of the data

    def __getitem__(self, idx):
        # takes each row 1 to end and turns it into a 28 x 28 matrix (row 0 is the labels ie pixel 0, pixel 1, etc)
        sample = self.data.iloc[idx].values.astype('uint8').reshape((28, 28))
        if self.transform:
            sample = self.transform(sample)
            # transform normalizes the data
        return sample

# create neural network
# new class that inherits from nn.Module
class numNN_train(nn.Module):
    # new initialization method from the new class
    def __init__(self):
        # call initialization method for the parent class nn.Module
        super().__init__()
        # creates weights with value of 0 to be trained
        self.w00 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.b00 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.w01 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # need to set requires_grad to true in order to train it

        self.w10 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.w11 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # final bias in this case is right before the output


    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = optim.relu(input_to_top_relu)
        scaled_top_relu_ouput = top_relu_output * self.w01
        
        input_to_bottom_relu = input * self.w10 + self.b10
        top_bottom_output = optim.relu(input_to_bottom_relu)
        scaled_bottom_relu_ouput = top_bottom_output * self.w11

        input_to_final_relu = scaled_top_relu_ouput + scaled_bottom_relu_ouput + self.final_bias

        output = optim.relu(input_to_final_relu)

        return output

# Load training data
# this composes several transforms together

transform = transforms.Compose([
    transforms.ToTensor(),
    # Convert to tensor
    # Normalize the pixel values, first number subtracts from original, second number divides
    # makes values between -1 and 1
    transforms.Normalize((.5,), (.5,))
])

# Construct the relative path to the CSV file
base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
csv_file_path = str(base_path) + r'\data\test.csv'  # Construct the relative path to the CSV file

# load data
number_dataset = numberDataset(csv_file=csv_file_path, transform=transform)

# loads in data 64 at a time and shuffles dataset for each epoch
dataloader = DataLoader(number_dataset, batch_size=64, shuffle=True)




# Train your model on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# Link to the description of this: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html