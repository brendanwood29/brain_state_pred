import sys
import json
from utils import get_train_loader
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from models import MLP
from torchinfo import summary


def get_config() -> ListConfig | DictConfig:
    
    default_config_path = "config/config.yaml"

    cfg = OmegaConf.load(default_config_path)

    if len(sys.argv) > 1:
        user_cfg = OmegaConf.load(sys.argv[1])
        cfg = OmegaConf.merge(cfg, user_cfg)

    return cfg


if __name__ == '__main__':
    
    config = get_config()
    with open('splits/train.json', 'r') as f:
        data = json.load(f)
    
    train_loader = get_train_loader('splits/train.json', 5)
    model = MLP(
        in_size=train_loader[0][0].shape[0],
        out_size=train_loader[0][1].shape[0],
        widths=(100, 100, 100)
    )
    
    for x, y in train_loader:
        
        y_hat = model(x)
        
        
        
        
        
        

    
