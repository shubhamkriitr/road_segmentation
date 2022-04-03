import torchvision.models as tvm
from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor
import torch
import logging
from torch import nn
import torch.functional as F
from loggingutil import logger

PRETAINED_MODEL_PATHS = {
    "torchvision": {
        "resnet50": "saved_models/torch/resnet50/resnet50-0676ba61.pth"
    }
}
# >>> resnet50 = tvm.resnet50(pretrained=True, progress=True)

def get_pruned_resnet_50 (load_strictly=False):
    model_weights_path = PRETAINED_MODEL_PATHS["torchvision"]["resnet50"]
    logger.debug(f"Loadin weights from : {model_weights_path}")
    state_dict = torch.load(model_weights_path)
    model = PrunedResnet50()
    missing_keys, unexpected_keys \
        = model.load_state_dict(state_dict=state_dict, strict=load_strictly)
    logger.debug(f"Missing Keys: {missing_keys}")
    logger.debug(f"Unexpected Keys: {unexpected_keys}")
    return model

class PrunedResnet50(ResNet):
    def __init__(self) -> None:
        # init params taken from : torchvision/models/resnet.py
        # "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])
        self.segmentation_output_channels = 1
        # new layers appended for segmentation (decoder part)
        self.transpose_block1 = self.make_transposed_block(512, 256)
        self.transpose_block2 = self.make_transposed_block(512, 64)
        self.transpose_block3 = self.make_transposed_block(128, 32)
        self.conv_final = nn.Conv2d(
            in_channels=32, out_channels=self.segmentation_output_channels,
            kernel_size=3, padding="same")
        self.final_activation_layer = nn.Sigmoid()
        self.remove_unnecessary_layers()
    
    def remove_unnecessary_layers(self):
        # refer forward function to see what needs to be removed
        # >>> x = self.layer3(x)
        # >>> x = self.layer4(x)
        # >>> x = self.avgpool(x)
        # >>> x = torch.flatten(x, 1)
        # >>> x = self.fc(x)
        del self.layer3
        del self.layer4
        del self.avgpool
        del self.fc
        
    def forward(self, x: Tensor) -> Tensor:
        #>>> return super()._forward_impl(x)
        #  N.B. : Make sure to check this implementation in parent class
        x = self.conv1(x)
        x = self.bn1(x)
        x_conv_bn_relu = self.relu(x)
        x = self.maxpool(x_conv_bn_relu)

        x_layer1 = self.layer1(x)
        x = self.layer2(x_layer1)
        
        # Skipping these steps (compared to original network)
        # >>> x = self.layer3(x)
        # >>> x = self.layer4(x)
        # >>> x = self.avgpool(x)
        # >>> x = torch.flatten(x, 1)
        # >>> x = self.fc(x)
        
        x = self.transpose_block1(x)
        x = torch.cat([x, x_layer1], dim=1) # concat along channels
        del x_layer1
        
        x = self.transpose_block2(x)
        x = torch.cat([x, x_conv_bn_relu], dim=1)
        del x_conv_bn_relu
        
        x = self.transpose_block3(x)
        
        x = self.conv_final(x)
        x = self.final_activation_layer(x)
        
        return x
    
    def make_transposed_block(self, in_channels, out_channels,
                              padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
                output_padding=output_padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
        

