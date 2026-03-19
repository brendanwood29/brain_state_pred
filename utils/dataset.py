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
    ):
        super().__init__()

        with open(split_path, 'r') as f:
            data = json.load(f)
        
        self.dataset = []
        self.bold = []
        self.weights = []
        self.edge_idxs = []
        
        scan_num = 0
        for subject in data:
            for ses in data[subject]:
                bold_data = pd.read_csv(data[subject][ses]['file_path'], index_col=0).to_numpy()
                fc = pd.read_csv(data[subject][ses]['file_path'].replace('cleaned-timeseries', 'connectome'), index_col=0).to_numpy()
                
                src, des = np.where(np.abs(fc) > threshold)
                edge_idx = np.stack([src, des])
                weights = fc[src, des]
                
                self.bold.append(
                    torch.tensor(bold_data, dtype=torch.float)
                )
                self.weights.append(
                    torch.tensor(np.abs(weights), dtype=torch.float)
                )
                self.edge_idxs.append(
                    torch.tensor(edge_idx, dtype=torch.long)
                )
                
                data_length = bold_data.shape[0]
                for i in range(data_length):
                    if (i + step) < data_length:
                        self.dataset.append(
                            (scan_num, i, i+step)
                        )
                scan_num += 1
                
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        ref = self.dataset[idx]

        return Data(
            x=self.bold[ref[0]][ref[1]:ref[2]].t(),
            edge_index=self.edge_idxs[ref[0]],
            edge_attr=self.weights[ref[0]].unsqueeze(-1),
            y=self.bold[ref[0]][ref[2]].unsqueeze(-1)
        )
        
