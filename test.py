import torch 

random_tensor = torch.rand((2, 3))
b = torch.tensor([4, 5, 6])
sum_tensor = random_tensor + b
print(sum_tensor)