import torch 
import torch.nn as nn
import torch.nn.functional as F 

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