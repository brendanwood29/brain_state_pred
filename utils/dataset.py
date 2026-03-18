import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Data


class SingleSubjectBrainFuncDataset(TorchDataset):
    def __init__(self, bold_data: str, step: int, device: str):
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
    
    
class BrainFuncDataset(TorchDataset):
    
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
                            torch.tensor(bold_data[i:i+step], dtype=torch.float).t().flatten().to(device)
                        )
                        self.outputs.append(
                            torch.tensor(bold_data[i+step], dtype=torch.float).t().to(device)
                        )
                
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
        
        
class BrainFuncGraphDataset(PyGDataset):
    
    def __init__(
        self,
        split_path: str,
        threshold: float,
        step: int,
        device: str
    ):
        super().__init__()

        with open(split_path, 'r') as f:
            data = json.load(f)
        
        self.items = []
        
        for subject in list(data.keys())[:1]:
            for ses in data[subject]:
                bold_data = pd.read_csv(data[subject][ses]['file_path'], index_col=0).to_numpy().astype(float)
                fc = pd.read_csv(data[subject][ses]['file_path'].replace('cleaned-timeseries', 'connectome'), index_col=0).to_numpy().astype(float)
                fc[np.eye(fc.shape[0]).astype(bool)] = 0
                src, des = np.where(abs(fc) > threshold)
                edge_idx = np.stack([src, des])
                weights = fc[src, des]
                
                
                data_length = bold_data.shape[0]
                for i in range(data_length):
                    if (i + step) < data_length:
                        
                        self.items.append(
                            Data(
                                x=torch.tensor(bold_data[i:i+step].T, dtype=torch.float),
                                edge_index=torch.tensor(edge_idx, dtype=torch.long),
                                edge_attr=torch.tensor(weights, dtype=torch.float),
                                y=torch.tensor(bold_data[step+i], dtype=torch.float)
                            ).to(device)
                        )
                
    def len(self):
        return len(self.items)

    def get(self, idx):
        return self.items[idx]
    