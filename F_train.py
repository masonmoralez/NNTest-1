import torch
import torch.nn as nn
import torch.optim as optim
from F_nnModel import numNN_train

def train_model(train_loader, test_loader, batch_size, learning_rate = 0.001, epochs = 30):
    num_Model = numNN_train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(num_Model.parameters(), lr=learning_rate)
    running_loss = 0

    for epoch in range(epochs):
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
            print(i)
            if i == batch_size:  # Print every batch
                print(f'Epoch {epoch + 1}, Batch Size {batch_size}, Loss: {running_loss / 100:.3f}')
            if i % 200 == 199:  # Print every 200 mini-batches
                print(f'Epoch {epoch + 1}, Batch Size {batch_size}, Loss: {running_loss / 100:.5f}')
                running_loss = 0.0
        print('batch done')

    return num_Model 