import torchvision.models as tvm
from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor
import torch
import logging

logger = logging.getLogger(name=__name__)

PRETAINED_MODEL_PATHS = {
    "torchvision": {
        "resnet50": "saved_models/torch/resnet50/resnet50-0676ba61.pth"
    }
}
# >>> resnet50 = tvm.resnet50(pretrained=True, progress=True)

def get_pruned_resnet_50 ():
    model_weights_path = PRETAINED_MODEL_PATHS["torchvision"]["resnet50"]
    logger.debug(f"Loadin weights from : {model_weights_path}")
    state_dict = torch.load(model_weights_path)
    model = PrunedResnet50()
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model

class PrunedResnet50(ResNet):
    def __init__(self) -> None:
        # init params taken from : torchvision/models/resnet.py
        # "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])
        
    def forward(self, x: Tensor) -> Tensor:
        #>>> return super()._forward_impl(x)
        #  N.B. : Make sure to check this implementation in parent class
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

