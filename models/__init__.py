from .mlp import MLP

def get_model(name: str, **kwargs):
        
    configured_models = [
        'mlp',
    ]
    
    if name not in configured_models:
        raise NotImplementedError(f'{name} is not a configured model, `name` must be one of {configured_models}')
    
    if name == 'mlp':
        return MLP(
            **kwargs
        )
        