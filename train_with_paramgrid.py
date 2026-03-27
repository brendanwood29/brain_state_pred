import sys
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
from sklearn.model_selection import ParameterGrid
from train import main
# from train_single_subject import main

def get_config() -> ListConfig | DictConfig:
    
    default_config_path = "configs/default_config.yaml"

    cfg = OmegaConf.load(default_config_path)

    if len(sys.argv) > 1:
        user_cfg = OmegaConf.load(sys.argv[1])
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg


if __name__ == '__main__':

# All subjects
    params_to_check = {
        'num_heads': [2, 12],
        'steps': [3, 5, 7],
        'num_blocks': [5, 10], 
    }
    print(len(ParameterGrid(params_to_check)))
    for i, grid in enumerate(ParameterGrid(params_to_check)):
        cfg = get_config()
        
        cfg.run_name = '_'.join([f'{x}-{y}' for x, y in grid.items()])
        

        cfg.model.kwargs.num_heads = grid['num_heads']
        cfg.model.kwargs.steps = grid['steps']
        cfg.model.kwargs.num_blocks = grid['num_blocks']
        main(cfg)
    




# Single Subject
    # params_to_check = {
    #     'num_heads': [2, 12],
    #     'steps': [3, 5, 7],
    #     'num_blocks': [1],
    #     'lr': [1e-3, 1e-5] 
    # }
    # print(len(ParameterGrid(params_to_check)))
    # for i, grid in enumerate(ParameterGrid(params_to_check)):
    #     cfg = get_config()
        
    #     cfg.run_name = '_'.join([f'{x}-{y}' for x, y in grid.items()])
        

    #     cfg.model.kwargs.num_heads = grid['num_heads']
    #     cfg.model.kwargs.steps = grid['steps']
    #     cfg.model.kwargs.num_blocks = grid['num_blocks']
    #     cfg.optim.lr = grid['lr']
    #     main(cfg)
    
    
    