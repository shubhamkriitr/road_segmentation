import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import collections

import os
import imageio

torch.manual_seed(42)

#TODO: move to GPU when available

class UNetLevel(torch.nn.Module):
    def __init__(self, layers, dim_coef, in_channels, out_channels):
        super().__init__()
        layers_list = []
        first_conv_in_channels = None
        try:
            UNetLevel.id_conter += 1
        except AttributeError:
            UNetLevel.id_conter = 0
        if dim_coef == 0.5:
            layers_list.append((f"L{UNetLevel.id_conter}_pool", torch.nn.MaxPool2d(kernel_size=2)))
        elif dim_coef == 2:
            layers_list.append((
                f"L{UNetLevel.id_conter}_transpose",
                torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            ))
            layers_list.append((f"L{UNetLevel.id_conter}_relu_initial", torch.nn.ReLU()))
        elif dim_coef != 1:
            raise Exception("UNetLevel: unsupported dimension change")
        for l in range(layers):
            layer_in_channels = in_channels if l == 0 else out_channels
            layers_list.append((
                f"L{UNetLevel.id_conter}_conv{l}",
                torch.nn.Conv2d(layer_in_channels, out_channels, kernel_size=3, padding="same")
            ))
            layers_list.append((f"L{UNetLevel.id_conter}_relu_{l}", torch.nn.ReLU()))
        self.layers = torch.nn.Sequential(collections.OrderedDict(layers_list))
    def forward(self, x, append=None):
        for idx, l in enumerate(self.layers):
            x = l(x)
            if idx == 0 and append is not None:
                x = torch.concat([x, append], axis=1) #TODO: crop instead of padding
        return x

class UNetEncoder(torch.nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = torch.nn.ModuleList(levels)
    def forward(self, x):
        outs = []
        for l in self.levels:
            x = l(x)
            outs.append(x)
        return x, outs

class UNetDecoder(torch.nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.levels = torch.nn.ModuleList(levels)
    def forward(self, x, residual_connections):
        for l, c in zip(self.levels, residual_connections):
            x = l(x, append=c)
        return x

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder([
            UNetLevel(layers=2, dim_coef=1, in_channels=3, out_channels=64),
            UNetLevel(layers=2, dim_coef=0.5, in_channels=64, out_channels=128),
            UNetLevel(layers=2, dim_coef=0.5, in_channels=128, out_channels=256),
            UNetLevel(layers=2, dim_coef=0.5, in_channels=256, out_channels=512),
            UNetLevel(layers=2, dim_coef=0.5, in_channels=512, out_channels=1024),
        ])
        self.decoder = UNetDecoder([
            UNetLevel(layers=2, dim_coef=2, in_channels=1024, out_channels=512),
            UNetLevel(layers=2, dim_coef=2, in_channels=512, out_channels=256),
            UNetLevel(layers=2, dim_coef=2, in_channels=256, out_channels=128),
            UNetLevel(layers=2, dim_coef=2, in_channels=128, out_channels=64),
        ])
        self.decoder_top = torch.nn.Sequential(collections.OrderedDict([ #TODO: consider using a single output channel like the paper
            ("1x1_conv", torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding="same")),
            #no softmax here, should be handled by the loss itself
        ]))
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
    def forward(self, x):
        x, outs = self.encoder(x)
        outs.pop(-1) #no resisual connection from the last encoder level
        outs.reverse()
        x = self.decoder(x, outs)
        x = self.decoder_top(x)
        return x
    def custom_train(self, x, y, epochs):
        writer = SummaryWriter("./logs")
        writer.add_graph(self, x)
        for e in range(epochs): #TODO: split to batches
            pred = self.forward(x)
            loss = self.loss(pred, y)
            print(f"epoch {e+1} loss before update: {loss}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"loss after epoch {epochs} update: {self.loss(self.forward(x), y)}")

img_dir, groundtruth_dir = "./data/training/images", "./data/training/groundtruth"
img_list = os.listdir(img_dir)
x_arrs = [np.array(imageio.imread(os.path.join(img_dir, img)))[:, :, :3] for img in img_list]
y_arrs = [np.array(imageio.imread(os.path.join(groundtruth_dir, img))) for img in img_list]
x = torch.tensor(np.transpose(np.stack(x_arrs), [0, 3, 1, 2]), dtype=torch.float32)
y = torch.tensor(np.stack(y_arrs), dtype=torch.float32) / 255

y = torch.stack([y, 1-y], dim=1)

epochs = 100
checkpoint_path = f"./model/model_{epochs}.pkl"
unet = UNet()
unet.custom_train(x[:1], y[:1], epochs) #TODO: train on more data than just one image
#torch.save(unet, checkpoint_path)

pred = unet(x[:1])[0]
pred = torch.where(pred[0] >= pred[1], 255 * torch.ones_like(pred[0]), torch.zeros_like(pred[0])).to(torch.uint8)

imageio.imwrite("./output_mask.png", pred)
