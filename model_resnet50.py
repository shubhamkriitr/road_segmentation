import torchvision.models as tvm
from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor
import torch
import logging
from torch import nn
import torch.functional as F
from loggingutil import logger
import commonutil
import os

PRETAINED_MODEL_PATHS = {
    "torchvision": {
        "resnet50": "saved_models/torch/resnet50/resnet50-0676ba61.pth"
    }
}
# >>> resnet50 = tvm.resnet50(pretrained=True, progress=True)

def download_model_if_not_available():
    model_path = PRETAINED_MODEL_PATHS["torchvision"]["resnet50"]
    model_dir = os.path.abspath(os.path.join(model_path, os.pardir))
    os.makedirs(model_dir, exist_ok=True)
    download_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    if not os.path.exists(model_path):
        logger.info(f"Downloading resnet50 weights")
        command = f"wget -c {download_url} -O {model_path}"
        logger.info(f"Running: `{command}`")
        os.system(command)

download_model_if_not_available()


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


class FrozenPrunedResnet50(PrunedResnet50):
    """Same as PrunedResnet50 but with frozen ResNet weights.
    Note: load_state_dict must be called for freezing the weights
    """
    def __init__(self) -> None:
        super().__init__()
    
    def load_state_dict(self, state_dict,
                        strict = True):
        # load all the params
        loading_key_info = super().load_state_dict(state_dict, strict)
        
        # now freeze the weights
        self._freeze_weights()
        
        return loading_key_info
    
    def _freeze_weights(self):
        for name, param in self.named_parameters():
            if self._should_freeze(name):
                param.requires_grad = False
    
    def _should_freeze(self, name: str):
        if name.startswith("layer"):
            return True
        if name.startswith("conv1"):
            return True
        if name.startswith("bn1"):
            return True
        
        return False
        

class SplitAndStichPrunedResnet50(PrunedResnet50):
    def __init__(self) -> None:
        super().__init__()
        self.split_factor = 2
        
    def preprocess_input(self, x: torch.Tensor):
        n = self.split_factor # num of slice along a given dimension
        w_i = x.shape[2]//n
        w_j = x.shape[3]//n
        s = (x.shape[0]*n*n, x.shape[1],
                              w_i, w_j)
        
        m = torch.zeros(size=s)
        
        # FIXME: find torch's inbuilt functions to unroll 
        # like this
        k = 0
        for p in range(x.shape[0]):
            for i in range(n): 
                for j in range(n):
                    m[k] = x[p, :, i*w_i:(i+1)*w_i, j*w_j:(j+1)*w_j]
                    k += 1
                    
        m = m.to(device=x.device)
        return m
    
    def postprocess_output(self, y):
        #>>> y = (B*n*n, 1, h//n, w//n)
        n = self.split_factor
        nsq = n*n
        w_i = y.shape[2]
        w_j = y.shape[3]
        h = w_i*n
        w = w_j*n
        assert y.shape[0] % nsq == 0
        B = y.shape[0]//nsq
        
        s = (B, y.shape[1], h, w)
        m = torch.zeros(size=s)
        
        k = 0
        for p in range(B):
            for i in range(n): 
                for j in range(n):
                    m[p, :, i*w_i:(i+1)*w_i, j*w_j:(j+1)*w_j] = y[k, :, : ,:]
                    k += 1
        m = m.to(y.device)
        return m
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess_input(x)
        x =  super().forward(x)
        x = self.postprocess_output(x)
        return x
             
    
# TODO: add in model factory once decided we are using tht
def model_getter ( model_class, load_strictly=False):
    def _getter():
        model_weights_path = PRETAINED_MODEL_PATHS["torchvision"]["resnet50"]
        logger.debug(f"Loadin weights from : {model_weights_path}")
        state_dict = torch.load(model_weights_path)
        model = model_class()
        missing_keys, unexpected_keys \
            = model.load_state_dict(state_dict=state_dict, strict=load_strictly)
        logger.debug(f"Missing Keys: {missing_keys}")
        logger.debug(f"Unexpected Keys: {unexpected_keys}")
        
        frozen_params, trainable_params\
            = commonutil.group_param_names_by_trainability(model)
        
        logger.info(f"Frozen parameters: {frozen_params}")
        logger.info(f"Trainable parameters: {trainable_params}")
        
        return model
    
    return _getter

get_pruned_resnet50 = model_getter(PrunedResnet50, False)
get_frozen_pruned_resnet50 = model_getter(FrozenPrunedResnet50, False)
get_split_pruned_resnet50 = model_getter(SplitAndStichPrunedResnet50, False)
