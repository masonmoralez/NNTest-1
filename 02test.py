import os
# allows us to do tensor math, calculations, and storage of data
import torch
# imports the neural networks classes and functions from pytorch
import torch.nn as nn

from F_train import training
from F_nnModel import numNN_train

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

pixels = 28 # adjustable data parameter

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

''' Uncomment this code for debugging (purpose is to see the labels and images plus their size)'''

'''# Print some samples from train_loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)
print('Labels:', labels)

# Optionally, you can print the images in a more readable format (for example, as numpy arrays)
print('Images:', images.numpy())'''
    
# Need to pick a manual seed for randomization of the weights and biases that we are using
torch.manual_seed(11) # 11 is just a random number (in this case I picked it because its Jalen Brunson's number). Can really be any number
model = numNN_train()

