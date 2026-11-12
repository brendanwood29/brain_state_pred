from pytorch_trainer import ModelGetter

from .graph_based import GCN  # , STGCN
from .lstm import LSTM
from .mamba import MambaModel, OldMamba
from .mlp import ANN_MLP, MLP
from .transformer import TransformerMambaModel, TransformerModel

npi_model_getter = ModelGetter(
    {
        "mlp": MLP,
        "npi_mlp": ANN_MLP,
        "transformer": TransformerModel,
        "combo": TransformerMambaModel,
        "gcn": GCN,
        # "stgcn": STGCN,
        "lstm": LSTM,
        "mamba": MambaModel,
        "old_mamba": OldMamba,
    }
)
