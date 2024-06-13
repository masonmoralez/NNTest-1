import os

# allows us to do tensor math, calculations, and storage of data
import torch

# imports the neural networks classes and functions from pytorch
import torch.nn as nn

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
        sample = self.data.iloc[idx, 1:].values.astype('uint8').reshape((28, 28))
        label = self.data.iloc[idx,0]
        if self.transform:
            sample = self.transform(sample)
            # transform normalizes the data
        return sample, label

# create neural network
# new class that inherits from nn.Module
class numNN_train(nn.Module):
    # new initialization method from the new class
    # creates and initializes the intial biases
    def __init__(self):
        # call initialization method for the parent class nn.Module
        super().__init__()
        
        ''' tested different initializations with the variables '''
        # # creates weights with value of 0 to be trained
        # self.w00 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.b00 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.w01 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # # need to set requires_grad to true in order to train it

        # self.w10 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.w11 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # self.w20 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.b20 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.w21 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # final bias in this case is right before the output

    # goes through the neural network by taking an input value and calculating the output value with the weights, biases, and activation functions
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        # input_to_top_relu = input * self.w00 + self.b00
        # top_relu_output = F.relu(input_to_top_relu)
        # scaled_top_relu_ouput = top_relu_output * self.w01
        
        # input_to_bottom_relu = input * self.w10 + self.b10
        # top_bottom_output = F.relu(input_to_bottom_relu)
        # scaled_bottom_relu_ouput = top_bottom_output * self.w11

        # input_to_final_relu = scaled_top_relu_ouput + scaled_bottom_relu_ouput + self.final_bias

        # output = F.relu(input_to_final_relu)

        # return output
    
    
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
csv_file_path = os.path.join(base_path,'train.csv')  # Construct the relative path to the CSV file
# load data
number_dataset = numberDataset(csv_file=csv_file_path, transform=transform)

# loads in data 64 at a time and shuffles dataset for each epoch
dataloader = DataLoader(number_dataset, batch_size=64, shuffle=True)

def train(epochs, input, model, optimizer):
    for epoch in range(epochs):
        total_loss = 0
        # indexing method for going through the code (to update back to original use the indexing method on inputs)
        # for i in range (len(input)):
        #     input_i = (input[i][0]).float
        #     label_i = (input[i][1]).float

        for inputs, labels in dataloader:
            # The second argument is -1. In PyTorch, using -1 in the view function tells PyTorch 
            # to infer the size of that dimension based on the total number of elements and the 
            # other specified dimensions. (why -1 is used)

            inputs = (inputs.view(inputs.size(0), -1)).float()  # Flatten inputs
            labels = (labels.view(-1,1)).float()

            outputs = model(inputs)

            # Sum of Squared residuals loss function 
            loss = torch.mean((outputs - labels) ** 2)

            # calculates the gradients for all tensors that require grad
            loss.backward()

            # adds loss
            total_loss += loss.item()

            # takes step based on loss
            optimizer.step()

            # zeros out derivatives
            optimizer.zero_grad()
            # breaks function if 
            if total_loss <= 0.001:
                print("Number of Steps", str(epoch))
                break

        print("Step: ", epoch + 1, "Loss: ", total_loss)



# creates training model
number_train_model = numNN_train()

optimizer = SGD(number_train_model.parameters(), lr=0.1)

train(50, number_dataset, number_train_model,optimizer)
