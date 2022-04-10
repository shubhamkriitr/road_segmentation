import os
import torch
from pathlib import Path
from datetime import datetime
# Generic utility code

PROJECTPATH = Path(__file__).parent.parent

# Shall we take device info from single source? #TODO
def resolve_device(key=None):
    """
    Returns `device name` to be used. (`key` will be used to 
    resolve device name based on usage group, in case different device 
    is required for diffrerent purposes. Currently it is
    being ignored)
    """
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    if key is not None:
        raise NotImplementedError()
    return default_device
    
    

def get_timestamp_str(granularity=1000):
    if granularity != 1000:
        raise NotImplementedError()
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_")

def count_number_of_params(model):
    return sum(p.numel() for p in model.parameters())
    
def count_number_of_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def group_param_names_by_trainability(model):
    frozen_params = []
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    return frozen_params, trainable_params

class BaseFactory(object):
    def __init__(self, config=None) -> None:
        self.config = {} if config is None else config 
        self.resource_map = self.config["resource_map"] if "resource_map" in \
            self.config else {}
    
    def get(self, resource_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        # currently not using config
        try:
            resource_class = self.resource_map[resource_name]
        except KeyError:
            raise KeyError(f"{resource_name} is not allowed. Please use one of"
                           f" these names: {list(self.resource_map.keys())}")
        
        if config is not None:
            return resource_class(config=config)
        
        return resource_class(*args_to_pass, **kwargs_to_pass)