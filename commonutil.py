import os
import torch
from pathlib import Path
from datetime import datetime
# Generic utility code

PROJECTPATH = Path(__file__).parent.parent

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