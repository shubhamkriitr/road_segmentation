from torchvision.models.convnext import ConvNeXt, CNBlockConfig
from torch import Tensor
import torch
import logging
from torch import nn
import torch.functional as F
from loggingutil import logger
import commonutil

PRETAINED_MODEL_PATHS = {
    "torchvision": {
        "resnet50": "../data/saved_models/torch/resnet50/resnet50-0676ba61.pth",
        "convnext_tiny": "../data/saved_models/torch/convnext/convnext_tiny-983f1562.pth",
        "convnext_base": "../data/saved_models/torch/convnext/convnext_base-6075fbad.pth"
    }
}

class PrunedConvnextTiny(ConvNeXt):
    def __init__(self) -> None:
        # Configuration for tine convnext
        super().__init__(block_setting=[
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ],
            stochastic_depth_prob=0.1)

        self.segmentation_output_channels = 1

        # new layers appended for segmentation (decoder part)
        self.transpose_block1 = self.make_transposed_block(768, 384, 4)
        self.transpose_block2 = self.make_transposed_block(384 * 2, 192)
        self.transpose_block3 = self.make_transposed_block(192 * 2, 96)
        self.transpose_block4 = self.make_transposed_block(96 * 2, 64)
        self.transpose_block5 = self.make_transposed_block(64, 32)

        self.conv_final = nn.Conv2d(
            in_channels=32, out_channels=self.segmentation_output_channels,
            kernel_size=3, padding="same")

        self.conv_sharp = nn.Conv2d(
            in_channels=3, out_channels=self.segmentation_output_channels,
            kernel_size=7, padding="same") # Motivation, have a layer that just learns to fix roads from the output image

        self.final_activation_layer = nn.Sigmoid()
        self.remove_unnecessary_layers()

    def remove_unnecessary_layers(self):
        del self.classifier

    def forward(self, x: Tensor) -> Tensor:
        # >>> return super()._forward_impl(x)
        #  N.B. : Make sure to check this implementation in parent class
        x_0 = self.features[0](x)  # -> torch.Size([1, 96, 100, 100])
        x_1 = self.features[1](x_0)  # -> torch.Size([1, 96, 100, 100])
        x_2 = self.features[2](x_1)  # -> torch.Size([1, 192, 50, 50])
        x_3 = self.features[3](x_2)  # -> torch.Size([1, 192, 50, 50])
        x_4 = self.features[4](x_3)  # -> torch.Size([1, 384, 25, 25])
        x_5 = self.features[5](x_4)  # -> torch.Size([1, 384, 25, 25])
        x_6 = self.features[6](x_5)  # -> torch.Size([1, 768, 12, 12])
        x_7 = self.features[7](x_6)  # -> torch.Size([1, 768, 12, 12])

        # Decode
        out = self.transpose_block1(x_7)  # -> torch.Size([1, 512, 25, 25])
        out = torch.cat([out, x_5], dim=1)  # concat along channels

        out = self.transpose_block2(out)  # -> torch.Size([1, 192, 50, 50])
        out = torch.cat([out, x_3], dim=1)

        out = self.transpose_block3(out)  # -> torch.Size([1, 64, 100, 100])
        out = torch.cat([out, x_1], dim=1)

        out = self.transpose_block4(out)  # -> torch.Size([1, 64, 200, 200])
        out = self.transpose_block5(out)  # -> torch.Size([1, 32, 400, 400])

        out = self.conv_final(out)  # -> torch.Size([1, 3, 400, 400])
        out = self.final_activation_layer(out)

        out = self.conv_sharp(out)
        out = self.final_activation_layer(out)

        return out

    def make_transposed_block(self, in_channels, out_channels, kernel_size=3,
                              padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )


class PrunedConvnextBase(ConvNeXt):
    def __init__(self) -> None:
        # Configuration for tine convnext
        super().__init__(block_setting= [
            CNBlockConfig(128, 256, 3),
            CNBlockConfig(256, 512, 3),
            CNBlockConfig(512, 1024, 27),
            CNBlockConfig(1024, None, 3),
                                            ],
                        stochastic_depth_prob=0.5)

        self.segmentation_output_channels = 1

        # new layers appended for segmentation (decoder part)
        self.transpose_block1 = self.make_transposed_block(1024, 512, 4)
        self.transpose_block2 = self.make_transposed_block(512 * 2, 256)
        self.transpose_block3 = self.make_transposed_block(256 * 2, 128)
        self.transpose_block4 = self.make_transposed_block(128 * 2, 64)
        self.transpose_block5 = self.make_transposed_block(64, 32)

        self.conv_final = nn.Conv2d(
            in_channels=32, out_channels=self.segmentation_output_channels,
            kernel_size=3, padding="same")

        self.conv_sharp = nn.Conv2d(
            in_channels=self.segmentation_output_channels, out_channels=self.segmentation_output_channels,
            kernel_size=7, padding="same") # Motivation, have a layer that just learns to fix roads from the output image

        self.final_activation_layer = nn.Sigmoid()
        self.remove_unnecessary_layers()

    def remove_unnecessary_layers(self):
        del self.classifier

    def forward(self, x: Tensor) -> Tensor:
        # >>> return super()._forward_impl(x)
        #  N.B. : Make sure to check this implementation in parent class
        x_0 = self.features[0](x)  # -> torch.Size([1, 128, 100, 100])
        x_1 = self.features[1](x_0)  # -> torch.Size([1, 128, 100, 100])
        x_2 = self.features[2](x_1)  # -> torch.Size([1, 256, 50, 50])
        x_3 = self.features[3](x_2)  # -> torch.Size([1, 256, 50, 50])
        x_4 = self.features[4](x_3)  # -> torch.Size([1, 512, 25, 25])
        x_5 = self.features[5](x_4)  # -> torch.Size([1, 512, 25, 25])
        x_6 = self.features[6](x_5)  # -> torch.Size([1, 1024, 12, 12])
        x_7 = self.features[7](x_6)  # -> torch.Size([1, 1024, 12, 12])

        # Decode
        out = self.transpose_block1(x_7)  # -> torch.Size([1, 512, 25, 25])
        out = torch.cat([out, x_5], dim=1)  # concat along channels

        out = self.transpose_block2(out)  # -> torch.Size([1, 192, 50, 50])
        out = torch.cat([out, x_3], dim=1)

        out = self.transpose_block3(out)  # -> torch.Size([1, 64, 100, 100])
        out = torch.cat([out, x_1], dim=1)

        out = self.transpose_block4(out)  # -> torch.Size([1, 64, 200, 200])
        out = self.transpose_block5(out)  # -> torch.Size([1, 32, 400, 400])

        out = self.conv_final(out)  # -> torch.Size([1, 3, 400, 400])
        out = self.final_activation_layer(out)

        out = self.conv_sharp(out)
        out = self.final_activation_layer(out)

        return out

    def make_transposed_block(self, in_channels, out_channels, kernel_size=3,
                              padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )


# TODO: add in model factory once decided we are using tht
def model_getter(model_class, load_strictly=False):
    def _getter():
        if model_class == PrunedConvnextTiny:
            model_weights_path = PRETAINED_MODEL_PATHS["torchvision"]["convnext_tiny"]
        elif model_class == PrunedConvnextBase:
            model_weights_path = PRETAINED_MODEL_PATHS["torchvision"]["convnext_base"]
        else:
            raise Exception("Not supported model_class")
        logger.debug(f"Loadin weights from : {model_weights_path}")
        state_dict = torch.load(model_weights_path)
        model = model_class()
        missing_keys, unexpected_keys \
            = model.load_state_dict(state_dict=state_dict, strict=load_strictly)
        logger.debug(f"Missing Keys: {missing_keys}")
        logger.debug(f"Unexpected Keys: {unexpected_keys}")

        frozen_params, trainable_params \
            = commonutil.group_param_names_by_trainability(model)

        logger.info(f"Frozen parameters: {frozen_params}")
        logger.info(f"Trainable parameters: {trainable_params}")

        return model

    return _getter


get_pruned_convnext_tiny = model_getter(PrunedConvnextTiny, False)
get_pruned_convnext_base = model_getter(PrunedConvnextBase, False)