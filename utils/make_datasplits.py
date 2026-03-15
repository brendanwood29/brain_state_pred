import json
import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_SPLIT = 0.8
TRAIN_SETS = [
    'hcp'
]
VAL_SETS = [
    'hcp'
]
TEST_SETS = [
]

def split_train_val(subjects):
    subjects = np.array(subjects)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(subjects))

    split_idx = int(len(subjects) * TRAIN_SPLIT)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    return subjects[train_idx], subjects[val_idx]
    

def split_by_subjects(data_dir, splits):
    train = {}
    val = {}
    test = {}
    
    for dataset in data_dir.glob('*.json'):
        
        name = dataset.name.removesuffix('.json')
        with open(dataset, 'r') as f:
            data = json.load(f)
        
        if name in TRAIN_SETS and name in TEST_SETS:
            raise ValueError(f'{name} in train and test sets')
        if name in VAL_SETS and name in TEST_SETS:
            raise ValueError(f'{name} in val and test sets')
        
        if name in TRAIN_SETS and name in VAL_SETS:
            train_subs, val_subs = split_train_val(list(data.keys()))
            train.update((sub, data[sub]) for sub in train_subs)
            val.update((sub, data[sub]) for sub in val_subs)
        elif name in TRAIN_SETS:
            train.update((sub, data[sub]) for sub in data.keys())
        elif name in VAL_SETS:
            val.update((sub, data[sub]) for sub in data.keys())
        else:
            test.update((sub, data[sub]) for sub in data.keys())
    
    with open(splits.joinpath('train.json'), 'w') as f:
        json.dump(train, f, indent=4)
    with open(splits.joinpath('val.json'), 'w') as f:
        json.dump(val, f, indent=4)
    with open(splits.joinpath('test.json'), 'w') as f:
        json.dump(test, f, indent=4)
        

def split_single_subject(data_path, train_proportion):
    
    bold_data = pd.read_csv(data_path, index_col=0).to_numpy()
    bold_data = bold_data[30:, :]
    data_length = bold_data.shape[0]
    
    return bold_data[:int(train_proportion * data_length), :], bold_data[int(train_proportion * data_length):, :]

    

if __name__ == '__main__':
    
    data_dir = Path('data_like-npi')
    splits = Path('splits')
    splits.mkdir(parents=True, exist_ok=True)
    
    split_by_subjects(data_dir, splits)
    

        