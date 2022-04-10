import torch
from commonutil import BaseFactory

# models
from model_resnet50 import (get_frozen_pruned_resnet50, get_pruned_resnet50)
from unet import UNet, BaselineUNet
from efficient_unet import EfficientUNet
from model_convnext import (get_pruned_convnext_small, get_pruned_convnext_tiny)

MODEL_NAME_TO_CLASS_OR_INTIALIZER_MAP = {
    "FrozenPrunedResnet50": get_frozen_pruned_resnet50,
    "PrunedResnet50": get_pruned_resnet50,
    "UNet": UNet,
    "BaselineUNet": BaselineUNet,
    "EfficientUNet": EfficientUNet,
    "PrunedConvnextTiny": get_pruned_convnext_tiny,
    "PrunedConvnextSmall": get_pruned_convnext_small
}

# For saved models 
MODEL_NAME_TO_WEIGHTS_PATH = {

}
class ModelFactory(BaseFactory):
    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.resource_map = MODEL_NAME_TO_CLASS_OR_INTIALIZER_MAP

    def get(self, model_name, config=None):
        # handle models 
        return super().get(model_name, config)

class TrainedModelFactory(ModelFactory):
    def __init__(self, config = {}) -> None:
        super().__init__()
        # if config has `model_name_to_weights_path`
        self.config = config
        if "model_name_to_weights_path" not in self.config:
            self.config["model_name_to_weights_path"] \
                = MODEL_NAME_TO_WEIGHTS_PATH

        self.model_weights_path = self.config["model_name_to_weights_path"]
    
    def get(self, model_name):
        model_class =  super().get(model_name)
        model_weights_path = self.model_weights_path[model_name]

        model: torch.nn.Module = model_class() # Assumes model does not need init params

        state_dict = torch.load(model_weights_path)
        if hasattr(model, "load_state_dict_for_eval"):
            model.load_state_dict_for_eval(state_dict=state_dict, strict=True)
        else:
            model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model

    def get_lazy_loader(self, model_name):
        return lambda : self.get(model_name)
    
    def load_from_location(self, model_name, model_weights_path):
        model_class =  super().get(model_name)

        model: torch.nn.Module = model_class() # Assumes model does not need init params

        state_dict = torch.load(model_weights_path)
        if hasattr(model, "load_state_dict_for_eval"):
            model.load_state_dict_for_eval(state_dict=state_dict, strict=True)
        else:
            model.load_state_dict(state_dict=state_dict, strict=True)
        # make sure to call model.eval() or model.train() based on the usage
        return model
        

if __name__ == "__main__":
    model_factory = TrainedModelFactory()
    model = model_factory.get("CnnWithResidualConnectionPTB")
    print(model)