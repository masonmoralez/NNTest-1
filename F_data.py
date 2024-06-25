import torch
import pandas as pd
from torch.utils.data import Dataset

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