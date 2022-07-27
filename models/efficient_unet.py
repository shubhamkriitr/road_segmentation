from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

"""This file contains the implementation of our EfficientNet-based model."""

class EfficientUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(EfficientUNet, self).__init__()

        self.enet = torchvision.models.efficientnet_b3(pretrained=True)
        self.update_enet = False

        self.encoder0 = EfficientUNet._block(3, 32, name="enc0")

        self.bottleneck = EfficientUNet._block(1536, 1536, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            1536, 136, kernel_size=2, stride=2, padding=1, output_padding=1
        )
        self.decoder4 = EfficientUNet._block(2*136, 136, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            136, 48, kernel_size=2, stride=2
        )
        self.decoder3 = EfficientUNet._block(2*48, 48, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            48, 32, kernel_size=2, stride=2
        )
        self.decoder2 = EfficientUNet._block(2*32, 32, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            32, 24, kernel_size=2, stride=2
        )
        self.decoder1 = EfficientUNet._block(2*24, 24, name="dec1")

        self.upconv0 = nn.ConvTranspose2d(
            24, 32, kernel_size=2, stride=2
        )
        self.decoder0 = EfficientUNet._block(2*32, 32, name="dec0")
        self.last_cov = nn.Conv2d(
            32, 1, kernel_size=1, stride=1
        )

    def forward(self, x):
        if self.update_enet:
            enc1 = self.enet.features[1](self.enet.features[0](x))
            enc2 = self.enet.features[2](enc1)
            enc3 = self.enet.features[3](enc2)
            enc4 = self.enet.features[5](self.enet.features[4](enc3))
            enc5 = self.enet.features[8](self.enet.features[7](self.enet.features[6](enc4)))
        else:
            with torch.no_grad():
                enc1 = self.enet.features[1](self.enet.features[0](x))
                enc2 = self.enet.features[2](enc1)
                enc3 = self.enet.features[3](enc2)
                enc4 = self.enet.features[5](self.enet.features[4](enc3))
                enc5 = self.enet.features[8](self.enet.features[7](self.enet.features[6](enc4)))

        bottleneck = self.bottleneck(enc5)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, self.encoder0(x)), dim=1)
        dec0 = self.decoder0(dec0)
        return torch.sigmoid(self.last_cov(dec0))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding="same",
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding="same",
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
