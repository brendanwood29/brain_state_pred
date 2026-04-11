# Development of Subject Specific AI Models for Functional Brain State Prediction

This codebase was used for the development of models related to the final project for MBP1413.

# Running instructions
1. Install `uv` using the link found [here](https://docs.astral.sh/uv/getting-started/installation/)
2. Create a virtual environment by running `uv venv`
3. Instal `torch==2.8.0` with the correct CUDA version for your system
4. Install `pytorch_trainer` a custom package for simplifying training PyTorch models, link found [here](https://github.com/brendanwood29/pytorch_trainer.git)
5. Run `uv sync` to install the other dependencies
6. To train a new single subject model, first prepare your data in a folder with the suffix `*timseries.csv` and run `uv run train_single_subject.py configs/single_subject_config.yaml`
7. To train a new pre-trained model, first prepare your data using the `prepare_data.py` and `make_datasplits.py` scripts, then using `uv run train.py configs/config.yaml` train a new model.

For any issues please contact [Brendan Wood](mailto:bwt.wood@mail.utoronto.ca)