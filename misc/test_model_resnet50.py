from models.model_resnet50 import PrunedResnet50, PRETAINED_MODEL_PATHS
import torch


state_dict = torch.load(PRETAINED_MODEL_PATHS["torchvision"]["resnet50"])
model = PrunedResnet50()
model.load_state_dict(state_dict=state_dict, strict=True)

