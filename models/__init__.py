from pytorch_trainer import ModelGetter

from .graph_based import GCN  # , STGCN
from .lstm import LSTM
from .mamba import MambaModel
from .mlp import ANN_MLP, MLP
from .transformer_based import TransformerModel

npi_model_getter = ModelGetter(
    {
        "mlp": MLP,
        "npi_mlp": ANN_MLP,
        "transformer": TransformerModel,
        "gcn": GCN,
        # "stgcn": STGCN,
        "lstm": LSTM,
        "mamba": MambaModel,
    }
)
