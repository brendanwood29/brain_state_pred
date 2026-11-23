import json
import warnings
from collections import defaultdict
from pathlib import Path


def make_metadata(dataset_path: Path, dataset_name: str):

    dataset = defaultdict(lambda: defaultdict(dict))

    for file in dataset_path.rglob("**/*timeseries.csv"):
        dataset[file.parts[-4]][file.parts[-3]]["file_path"] = str(file)
        dataset[file.parts[-4]][file.parts[-3]]["tr"] = (
            0.72  # FIXME Make this read the tr from the json file, for now this is
        )
        warnings.warn(
            "USING HARDCODED TR ENSURE TR IS CORRECT",
            category=UserWarning,
            stacklevel=2,
        )

    with open(dataset_path.parent.joinpath(f"{dataset_name}.json"), "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    datasets = {
        # 'hcp': 'data_100p/hcp'
        # "hcp": "data/data_400p/hcp"
        # "aabc": "data/data_400p/aabc"
        # "rnn": "data/data_400p/rnn",
        "wbm": "data/data_66p/wbm_sim",
    }

    for name, path in datasets.items():
        make_metadata(Path(path), name)
