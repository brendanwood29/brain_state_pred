# Development of Subject Specific AI Models for Functional Brain State Prediction

This codebase was used for the development of models related to the final project for MBP1413.

# Running instructions
1. Install `uv` using the link found [here](https://docs.astral.sh/uv/getting-started/installation/). If you already have `uv` installed, ensure it is up to date (`v.0.11.16`)by running `uv self update`.
2. Create a virtual environment by running `uv venv`
3. Instal `torch==2.7.0` with the correct CUDA version for your system
```bash 
uv pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```
4. Install `pytorch_trainer` a custom package for simplifying training PyTorch models, link found [here](https://github.com/brendanwood29/pytorch_trainer.git)
5. Run `uv sync` to install the other dependencies
6. To train a new single subject model, first prepare your data in a folder with the suffix `*timseries.csv` and run `uv run train_single_subject.py configs/single_subject_config.yaml`
7. To train a new pre-trained model, first prepare your data using the `prepare_data.py` and `make_datasplits.py` scripts, then using `uv run train.py configs/config.yaml` train a new model.

For any issues please contact [Brendan Wood](mailto:bwt.wood@mail.utoronto.ca)

# Dev Notes
See [this](https://github.com/state-spaces/mamba/issues/842) post about installing Mamba
easiest way right now is to run (after the environment has been created)
```bash
    uv pip install https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu11torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl --no-cache-dir --no-deps
```
Then install the extra dependencies independently (should be `einops`, `huggingface_hub`, and `transformers`)
