import torch
import json
import pandas as pd
from torch.utils.data import Dataset


class BrainFuncDataset(Dataset):
    
    def __init__(self, split_path: str, step: int):
        super().__init__()
        
        self.inputs = []
        self.outputs = []
        with open(split_path, 'r') as f:
            data = json.load(f)
        
        for subject in data: # TODO make this handle longitudinal data, or figure out a good way to deal with it
            for ses in data[subject]: 
                bold_data = pd.read_csv(data[subject][ses]['file_path']).to_numpy()
                data_length = bold_data.shape[0]
                for i in range(data_length):
                    if (i + step) < data_length:
                        self.inputs.append(
                            torch.tensor(bold_data[i:i+step]).t().reshape(-1, 1)
                        )
                        self.outputs.append(
                            torch.tensor(bold_data[i+step]).t()
                        )
                
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
        
        
