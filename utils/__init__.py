from pathlib import Path

from torch_geometric.loader import DataLoader as PyGDataLoader

from pytorch_trainer import LossGetter

from .dataset import (
    BrainFuncGCNDataset,
    BrainFuncRecursiveDataset,
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
    "BrainFuncRecursiveDataset",
    "split_single_subject",
    "get_loss_fn",
    "evaluate_on_train_end",
]


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
