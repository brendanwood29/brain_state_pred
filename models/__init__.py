from .mlp import MLP, ANN_MLP
from .transformer_based import TransformerModel
from .graph_based import GCN, STGCN

def get_model(name: str, **kwargs):
        
    configured_models = [
        'mlp',
        'npi_mlp',
        'transformer',
        'gcn',
        'stcgn'
    ]
    
    if name not in configured_models:
        raise NotImplementedError(f'{name} is not a configured model, `name` must be one of {configured_models}')
    
    if name == 'mlp':
        return MLP(
            **kwargs
        )
    elif name == 'npi_mlp':
        return ANN_MLP(
            **kwargs
        )
    elif name == 'transformer':
        return TransformerModel(
            **kwargs
        )
    elif name == 'gcn':
        return GCN(
            **kwargs
        )
    elif name == 'stgcn':
        return STGCN(
            **kwargs
        )
        