import sys
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


        
        
        
        
        

    
