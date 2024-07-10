import os
# allows us to do tensor math, calculations, and storage of data
import torch
# imports the neural networks classes and functions from pytorch
import torch.nn as nn

from F_train import train_model

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
#import matplotlib.pyplot as plt

# data exploration
import pandas as pd

# Need to pick a manual seed for randomization of the weights and biases that we are using
torch.manual_seed(11) # 11 is just a random number (in this case I picked it because its Jalen Brunson's number). Can really be any number

pixels = 28

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
    def __init__(self, in_features = (pixels ** 2), h1 = 128, h2 = 64, out_features = 10):
        # call initialization method for the parent class nn.Module
        super(numNN_train, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    # goes through the neural network by taking an input value and calculating the output value with the weights, biases, and activation functions
    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flattens the input 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
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
''' We can make our data base set easier to use with a URL. (However, this is a problem that we can change later) '''
''' We also need to look at pandas and loading the data '''

base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
csv_file_path_train = os.path.join(base_path,'train.csv')
csv_file_path_test = os.path.join(base_path,'test.csv')  # Construct the relative path to the CSV file
# load data
number_dataset_train = numberDataset(csv_file=csv_file_path_train, transform=transform)
number_dataset_test = numberDataset(csv_file=csv_file_path_test, transform=transform)

# loads in data 64 at a time and shuffles dataset for each epoch
train_loader = DataLoader(number_dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(number_dataset_test, batch_size=64, shuffle=True)

# creates training model
number_train_model = numNN_train()

optimizer = SGD(number_train_model.parameters(), lr=0.01)

train_model(100, number_train_model,optimizer,train_loader,test_loader)

