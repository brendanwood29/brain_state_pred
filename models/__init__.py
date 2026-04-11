from .mlp import MLP, ANN_MLP
from .lstm import LSTM
from .transformer_based import TransformerModel
from .graph_based import GCN, STGCN
from pytorch_trainer import ModelGetter


npi_model_getter = ModelGetter(
    {
        'mlp': MLP,
        'npi_mlp':ANN_MLP,
        'transformer': TransformerModel,
        'gcn': GCN,
        'stgcn': STGCN,
        'lstm': LSTM
    }
)

