import os
# allows us to do tensor math, calculations, and storage of data
import torch
# imports the neural networks classes and functions from pytorch
import torch.nn as nn

from F_train import training
from F_nnModel import numNN_train
from F_data import CustomDigitDataset

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

base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
train_dataset = CustomDigitDataset(os.path.join(base_path,'train.csv'))
test_dataset = CustomDigitDataset(os.path.join(base_path,'test.csv'))
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True)
    # - train_dataset (the dataset that we loaded)
    # - batch_size (how many samples will be passed through the data at one time)
    # - shuffle (enabling this with True means that you get random data each time)

''' Uncomment this code for debugging (purpose is to see the labels and images plus their size)'''

# Print some samples from train_loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

print('Images shape:', images.shape)
print('Labels shape:', labels.shape)
print('Labels:', labels)

# Optionally, you can print the images in a more readable format (for example, as numpy arrays)
print('Images:', images.numpy())

def train_model(train_loader, test_loader, learning_rate = 0.001, epochs = 30):
    num_Model = numNN_train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(num_Model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0
        early_stop = False
        for i, data in enumerate(train_loader,0):
        # this iterates through train_loader, starting at a 
            inputs, labels = data
            # Each data yielded by train_loader is a tuple containing a batch of inputs and their corresponding labels
            # this simply extracts the inputs and labels from data, in a index value format
            optimizer.zero_grad() # Process of zeroing the gradients
            # Forward pass
            outputs = num_Model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()
            # Accumulate loss
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.7f}')
                if running_loss / 100 <= 0.001:
                    print(f"Number of Steps {epoch + 1}")
                    early_stop = True
                    break
                running_loss = 0.0
        if early_stop:
            break
        elif epoch + 1 == epochs:
            print(f"Finished Training Final Loss:  {running_loss}")

# Need to pick a manual seed for randomization of the weights and biases that we are using
torch.manual_seed(11) # 11 is just a random number (in this case I picked it because its Jalen Brunson's number). Can really be any number
train_model(train_loader, test_loader)