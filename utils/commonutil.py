import os
import torch
from pathlib import Path
from datetime import datetime
import imageio
import numpy as np
# Generic utility code
from utils.loggingutil import logger

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

def to_cuda_if_available(pytorch_object):
    try:
        return pytorch_object.cuda()
    except RuntimeError:
        return pytorch_object

def write_images(model, dataloader, path, threshold,
                 save_submission_files=False, offset=144):
    if not os.path.exists(path): os.makedirs(path)
    submit_dir = os.path.join(path, "../submit", "predictions")
    sample_index = -1
    if save_submission_files:
        os.makedirs(submit_dir)
        logger.info(f"len(dataloader.dataset) = {len(dataloader.dataset)}")
    for b, (x, y) in enumerate(dataloader):
        pred = None
        with torch.no_grad():
            if next(model.parameters()).is_cuda:
                x = x.cuda()
            pred = model(x).cpu()
        thresholded = torch.where(pred >= threshold, torch.ones_like(pred), torch.zeros_like(pred)).numpy()
        pred = pred.numpy()
        y = y.cpu().numpy()[:, :1]
        assert pred.shape == y.shape
        for i in range(pred.shape[0]):
            sample_index += 1
            imageio.imwrite(f"{path}/{b}-{i}.png", (pred[i, 0]*255).astype(np.uint8))
            imageio.imwrite(f"{path}/{b}-{i}_thresholded.png", (thresholded[i, 0]*255).astype(np.uint8))
            imageio.imwrite(f"{path}/{b}-{i}_label.png", (y[i, 0]*255).astype(np.uint8))
            if save_submission_files:
                image_filename = f"satimage_{offset+sample_index}.png"
                logger.info(f"Saving image: {image_filename}")
                imageio.imwrite(f"{submit_dir}/{image_filename}",
                                (thresholded[i, 0]*255).astype(np.uint8))
            
            
            
class BaseFactory(object):
    def __init__(self, config=None) -> None:
        self.config = {} if config is None else config 
        self.resource_map = self.config["resource_map"] if "resource_map" in \
            self.config else {}
    
    def get(self, resource_name, config=None,
            args_to_pass=[], kwargs_to_pass={}):
        # currently not using config
        resource_class = self.get_uninitialized(resource_name)
        
        if config is not None:
            return resource_class(config=config)
        
        return resource_class(*args_to_pass, **kwargs_to_pass)

    def get_uninitialized(self, resource_name):
        try:
            return self.resource_map[resource_name]
        except KeyError:
            raise KeyError(f"{resource_name} is not allowed. Please use one of"
                           f" these names: {list(self.resource_map.keys())}")

def read_config(config_path):
    config_data = None
    with open(config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    assert config_data is not None, "Config file not found"
    return config_data
