import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

STRICT_TRAIN_TEST = (
    False  # If true, prevents training and testing having the same datasets
)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TRAIN_SETS = ["hcp"]
VAL_SETS = [""]
TEST_SETS = [""]


def split_train_val_test(subjects):
    subjects = np.array(subjects)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(subjects))

    tr_split = int(len(subjects) * TRAIN_SPLIT)
    val_split = int(len(subjects) * VAL_SPLIT) + tr_split
    train_idx = indices[:tr_split]
    val_idx = indices[tr_split:val_split]
    test_idx = indices[val_split:]

    return subjects[train_idx], subjects[val_idx], subjects[test_idx]


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

    for dataset in data_dir.glob("*.json"):
        if "aabc" in dataset.name:  # or "hcp" in dataset.name:
            continue
        name = dataset.name.removesuffix(".json")
        with open(dataset, "r") as f:
            data = json.load(f)

        if STRICT_TRAIN_TEST:
            if name in TRAIN_SETS and name in TEST_SETS:
                raise ValueError(f"{name} in train and test sets")
            if name in VAL_SETS and name in TEST_SETS:
                raise ValueError(f"{name} in val and test sets")

        if name in TRAIN_SETS and name in VAL_SETS and name in TEST_SETS:
            train_subs, val_subs, test_subs = split_train_val_test(list(data.keys()))
            train.update((sub, data[sub]) for sub in train_subs)
            val.update((sub, data[sub]) for sub in val_subs)
            test.update((sub, data[sub]) for sub in test_subs)
        elif name in TRAIN_SETS and name in VAL_SETS:
            train_subs, val_subs = split_train_val(list(data.keys()))
            train.update((sub, data[sub]) for sub in train_subs)
            val.update((sub, data[sub]) for sub in val_subs)
        elif name in TRAIN_SETS:
            train.update((sub, data[sub]) for sub in data.keys())
        elif name in VAL_SETS:
            val.update((sub, data[sub]) for sub in data.keys())
        else:
            test.update((sub, data[sub]) for sub in data.keys())

    if splits.joinpath("train.json").is_file():
        raise FileExistsError("train.json exists!")
    with open(splits.joinpath("train.json"), "w") as f:
        json.dump(train, f, indent=4)

    if splits.joinpath("val.json").is_file():
        raise FileExistsError("val.json exists!")
    with open(splits.joinpath("val.json"), "w") as f:
        json.dump(val, f, indent=4)

    if splits.joinpath("test.json").is_file():
        raise FileExistsError("test.json exists!")
    with open(splits.joinpath("test.json"), "w") as f:
        json.dump(test, f, indent=4)


def resample_bold(data: np.ndarray, tr: float, target_tr: float):

    ratio = Fraction(tr / target_tr).limit_denominator(10000)

    return resample_poly(
        data, ratio.numerator, ratio.denominator, axis=0, padtype="line"
    )


def split_single_subject(
    data_path: str | Path, train_proportion: float, tr: float, target_tr: float | None
):

    bold_data = pd.read_csv(data_path, index_col=0).to_numpy()
    # exclude non-steady state (first 30 seconds to be safe)
    to_clip = np.ceil(30 / tr).astype(int)
    bold_data = bold_data[to_clip:, :]

    if target_tr is not None:
        bold_data = resample_bold(bold_data, tr, target_tr)

    data_length = bold_data.shape[0]

    boundary = np.ceil(train_proportion * data_length).astype(int)
    return bold_data[:boundary, :], bold_data[boundary:, :]


if __name__ == "__main__":
    data_dir = Path("data/data_like-npi")
    splits = Path("splits/splits_like-npi")
    splits.mkdir(parents=True, exist_ok=True)

    split_by_subjects(data_dir, splits)
