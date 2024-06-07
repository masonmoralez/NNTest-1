import os
import torch
# imports the neural networks classes and functions from pytorch
import torch.nn as nn
# imports the optimization algorithms from pytorch like stochastic gradient descrent
import torch.optim as optim
# classes for handling datasets and creating data loaders
from torch.utils.data import Dataset,DataLoader
# allows you to normalize your data
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
class numberDataset(Dataset):
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

# Load training data
transform = transforms.Compose([
    # Convert to tensor
    transforms.ToTensor(),
    # Normalize the pixel values, first number subtracts from original, second number divides
    # makes values between -1 and 1
    transforms.Normalize((.5,), (.5,))
])

# Construct the relative path to the CSV file
base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
csv_file_path = str(base_path) + r'\data\test.csv'  # Construct the relative path to the CSV file

# load data
number_dataset = numberDataset(csv_file=csv_file_path,transform=transform)

# loads in data 64 at a time and shuffles dataset for each epoch
dataloader = DataLoader(number_dataset, batch_size=64, shuffle=True)

# create neural network
# class numberModel(nn.Module):
#     def __init__(self):
#         super().__init__()

# Train your model on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
