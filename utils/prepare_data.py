import json
from collections import defaultdict
from pathlib import Path

def make_metadata(dataset_path: Path, dataset_name: str):

    dataset = defaultdict(lambda: defaultdict(dict))
    
    for file in dataset_path.rglob('**/*timeseries.csv'):
        dataset[file.parts[-4]][file.parts[-3]]['file_path'] = str(file)
        dataset[file.parts[-4]][file.parts[-3]]['tr'] = 2 #FIXME Make this read the tr from the json file, for now this is fine since we are not using it
    
    with open(dataset_path.parent.joinpath(f'{dataset_name}.json'), 'w') as f:
        json.dump(dataset, f, indent=4)
    

if __name__ == '__main__':
    
    datasets = {
        # 'adni': '/data3/projects/bwood/classes/brain_state_pred/toy_data/adni'
        'hcp': 'data_like-npi/hcp'
    }
    
    for name, path in datasets.items():
    
        make_metadata(
            Path(path),
            name
        )