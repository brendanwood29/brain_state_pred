import torch
import json
import pandas as pd
from torch.utils.data import Dataset


class SingleSubjectBrainFuncDataset(Dataset):
    def __init__(self, bold_data, step, device):
        super().__init__()
        self.inputs = []
        self.outputs = []
        data_length = bold_data.shape[0]
        for i in range(data_length):
            if (i + step) < data_length:
                self.inputs.append(
                    torch.tensor(bold_data[i:i+step], dtype=torch.float).flatten().to(device)
                )
                self.outputs.append(
                    torch.tensor(bold_data[i+step], dtype=torch.float).t().to(device)
                )
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
    
    
class BrainFuncDataset(Dataset):
    
    def __init__(self, split_path: str, step: int, device: str):
        super().__init__()
        
        self.inputs = []
        self.outputs = []
        with open(split_path, 'r') as f:
            data = json.load(f)
        
        for subject in data: # TODO make this handle longitudinal data, or figure out a good way to deal with it
            for ses in data[subject]: 
                bold_data = pd.read_csv(data[subject][ses]['file_path'], index_col=0).to_numpy()
                data_length = bold_data.shape[0]
                for i in range(data_length):
                    if (i + step) < data_length:
                        self.inputs.append(
                            torch.tensor(bold_data[i:i+step], dtype=torch.float).t().reshape(-1).to(device)
                        )
                        self.outputs.append(
                            torch.tensor(bold_data[i+step], dtype=torch.float).t().to(device)
                        )
                
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
        
        
