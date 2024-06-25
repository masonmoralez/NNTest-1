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
import matplotlib.pyplot as plt

# data exploration
import pandas as pd

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

def train_model(train_loader, test_loader, learning_rate = 0.01, epochs = 10):
    model = numNN_train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for i, data 
    return 

# Need to pick a manual seed for randomization of the weights and biases that we are using
torch.manual_seed(11) # 11 is just a random number (in this case I picked it because its Jalen Brunson's number). Can really be any number


import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the advanced neural network class (as previously defined)
class AdvancedNN(nn.Module):
    def __init__(self):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # First hidden layer with 256 nodes
        self.fc2 = nn.Linear(256, 128)      # Second hidden layer with 128 nodes
        self.fc3 = nn.Linear(128, 64)       # Third hidden layer with 64 nodes
        self.fc4 = nn.Linear(64, 10)        # Output layer with 10 nodes
        self.dropout = nn.Dropout(0.5)      # Dropout layer for regularization

    def forward(self, x):
        x = x.view(-1, 28 * 28)            # Flatten the input
        x = torch.relu(self.fc1(x))        # Apply ReLU to first hidden layer
        x = self.dropout(x)                # Apply dropout
        x = torch.relu(self.fc2(x))        # Apply ReLU to second hidden layer
        x = self.dropout(x)                # Apply dropout
        x = torch.relu(self.fc3(x))        # Apply ReLU to third hidden layer
        x = self.fc4(x)                    # Output layer (no activation here, will use CrossEntropyLoss)
        return x

# Function to train the advanced network
def train_network_advanced(train_loader, test_loader, learning_rate=0.01, epochs=10):
    net = AdvancedNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print(f"Training Advanced Network with learning rate: {learning_rate}")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
        # enumeratare(,) creates an enumerate object that yields pairs of (index, value) starting from zero
            # - i is the index
            # - data is the value 
            inputs, labels = data


    print('Finished Training Advanced Network\n')

    # Calculate and print accuracy on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

train_network_advanced(train_loader, test_loader)

