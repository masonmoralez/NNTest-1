import torch
import torch.nn as nn
import torch.optim as optim

def training(epochs,model, optimizer,dataloader,testloader):
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

        print("Step: ", epoch + 1, "Loss: ", total_loss, ", ", loss)