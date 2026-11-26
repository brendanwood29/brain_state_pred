import json
from pathlib import Path

import torch

from evaluation import evaluate_on_train_end
from models import npi_model_getter
from utils import (
    fix_seeds,
    get_config,
    get_sub_info,
    load_trained_model,
    split_single_subject,
)


def main(cfg):

    # inputs
    label_file = "data/utils/region_labels.txt"
    json_file = "splits/splits_like-npi/train.json"

    with open(json_file, "r") as f:
        data = json.load(f)

    subject = get_sub_info(cfg.run_name, data)
    tr = subject["tr"]
    data_path = Path(subject["file_path"])
    _, test_data = split_single_subject(
        data_path, cfg.data.train_proportion, tr, cfg.data.target_tr
    )
    run = Path(cfg.work_dir).joinpath(cfg.run_name)

    model = npi_model_getter(cfg.model.name, **cfg.model.kwargs)
    model = load_trained_model(model, run.joinpath("models", "final_model.pt"))

    with torch.no_grad():
        evaluate_on_train_end(
            cfg=cfg,
            test_data=torch.tensor(test_data),
            model=model,
            label_file=label_file,
            device=cfg.device,
        )


if __name__ == "__main__":
    fix_seeds(42)
    cfg = get_config()
    main(cfg)
