from pathlib import Path

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from pytorch_trainer import LossGetter

from .dataset import (
    BrainFuncDataset,
    BrainFuncGCNDataset,
    SingleSubjectBrainFuncDataset,
    SingleSubjectBrainFuncGCNDataset,
    SingleSubjectBrainFuncRecursiveDataset,
    SingleSubjectBrainFuncSTGCNDataset,
    SingleSubjectFFTFuncDataset,
)
from .evaluate import evaluate_on_train_end
from .loss_fns import MSE1DLoss, MSEFCLoss, RealImagMSE, ReconFourierLoss
from .make_datasplits import split_single_subject

__all__ = [
    "SingleSubjectBrainFuncDataset",
    "SingleSubjectBrainFuncRecursiveDataset",
    "SingleSubjectBrainFuncGCNDataset",
    "SingleSubjectBrainFuncSTGCNDataset",
    "SingleSubjectFFTFuncDataset",
    "split_single_subject",
    "get_loss_fn",
    "evaluate_on_train_end",
]


def get_loader(
    data_path: str | Path,
    step: int,
    strength: float,
    batch_size=64,
    shuffle=True,
    **kwargs,
):

    return TorchDataLoader(
        BrainFuncDataset(data_path, step, strength),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )


def get_pyg_loader(
    data_path: str | Path,
    threshold: float,
    step: int,
    batch_size=64,
    shuffle=True,
    **kwargs,
):

    return PyGDataLoader(
        BrainFuncGCNDataset(data_path, threshold, step),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )


get_loss_fn = LossGetter(
    {
        "real_img_loss": RealImagMSE,
        "mse_fc_loss": MSEFCLoss,
        "mse_fourier_loss": ReconFourierLoss,
        "mse_first_loss": MSE1DLoss,
    }
)
