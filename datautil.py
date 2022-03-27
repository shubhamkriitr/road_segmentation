from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import torchvision.transforms.functional as TF
import torchvision
import torch
import logging
import PIL
import numpy as np
import random

# TODO: better to take this from run config sigleton
TENSOR_FLOAT_DTYPE = torch.float32

# Using prefix `Seg` (for segmentation) to the classes below to distinguish
# them from torchvision's  classes
class SegRotationTransform:
    def __init__(self, angles=None):

        self.angles = angles
        if self.angles is None:
            self.angles = [30, 60, 90, 120, 180]

    def __call__(self, x, y):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle), TF.rotate(y, angle)

class SegHorizontalFlip:
    def __init__(self, *args, **kwargs) -> None:
        #no attrs
        pass

    def __call__(self, x, y):
        return TF.hflip(x), TF.hflip(y)


class SegVerticalFlip:
    def __init__(self, *args, **kwargs) -> None:
        #no attrs
        pass

    def __call__(self, x, y):
        return TF.vflip(x), TF.vflip(y)
    
class IdentityTransform:
    def __init__(self, *args, **kwargs) -> None:
        #no attrs
        pass

    def __call__(self, x, y):
        return x, y

class SegAdjustBrightness:
    def __init__(self, brightness_factors=None) -> None:
        #no attrs
        self.brightness_factors = brightness_factors
        if self.brightness_factors is None:
            self.brightness_factors = [0.5, 0.8, 1.2, 1.5]

    def __call__(self, x, y):
        factor = random.choice(self.brightness_factors)
        return TF.adjust_brightness(x, brightness_factor=factor), y


default_transformations = {IdentityTransform(): 0.4,
                           SegHorizontalFlip(): 0.2,
                          SegVerticalFlip(): 0.2,
                          SegAdjustBrightness(): 0.2}

        
class CILRoadSegmentationDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 image_folder = "images",
                 label_folder = "groundtruth", # Set to none if no supervision
                 transformations = default_transformations # Dictionary where key is class to be applied, value is the probability
        ):
        
        self.root_dir = root_dir
        self.num_samples = 0
        self.image_names = []
        
        # Define paths for both images and labels given the root directory
        self.images_path = os.path.join(self.root_dir, image_folder)
        self.labels_path = os.path.join(self.root_dir, label_folder) if label_folder else None
        
        # Store transformations
        self.transformations = transformations
        
        # Store name of images
        self._inspect_rootdir()
        
    def _inspect_rootdir(self):
        is_img = lambda x: '.png' in x
        self.image_names = [f for f in os.listdir(self.images_path) if is_img(f)]
        self.num_samples = len(self.image_names)
        
        if self.labels_path:
            assert self.num_samples == sum(is_img(filename) for filename in os.listdir(self.labels_path)), "Number of images does not match number of labels"
            
        if self.transformations:
            assert np.array(list(self.transformations.values())).sum() == 1, "Probabilities of transformations should up to 1"
        
        assert self.num_samples > 0, "No input data was found"
        
    def load_image(self, image_path):
        # N.B.: img_array is of unit8 dtype with pixel value range [0-255]
        # normalize it later in the pipeline or store normalized version
        img_array = np.asarray(PIL.Image.open(image_path))
        return img_array

    def to_tensor(self, img_array):
        return torch.tensor(data=img_array, dtype=TENSOR_FLOAT_DTYPE)
    
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        input_image = self.load_image(os.path.join(self.images_path, self.image_names[index]))
        
        # torchvision requires image in [0, 1) range for float dtype
        input_image = np.clip(input_image/255, a_min=0, a_max= 1 - 1e-7)
        
        if self.labels_path:
            groundtruth = self.load_image(os.path.join(self.labels_path, self.image_names[index]))

            # NOTE: creating probability map from the ground truth image
            groundtruth = groundtruth//255
        else:
            groundtruth = None
        
        # To tensors
        input_image = self.to_tensor(input_image).transpose(1, 2).transpose(0, 1)
        groundtruth = self.to_tensor(groundtruth).unsqueeze(0) if groundtruth is not None else None
        
        # remove unnecessary dim
        input_image = input_image[:3]
        
        if self.transformations and self.labels_path:
            # Get transform according to probabilities
            transform = np.random.choice(list(self.transformations.keys()), size=1, p=list(self.transformations.values()))[0]
            input_image, groundtruth = transform(input_image, groundtruth)

        return input_image[:3], groundtruth
    

def get_dataset(root_dir: str,
                transformations = "default",
                 image_folder = "images",
                 label_folder = "groundtruth"
                ):
    
    transformations = default_transformations if transformations=="default" else transformations if type(transformations) is dict else None
    return CILRoadSegmentationDataset(root_dir=root_dir,
                                     transformations=transformations,
                                    image_folder = image_folder,
                                     label_folder = label_folder)

def get_dataloader(root_dir: str,
                                                  batch_size: int = 10,
                               shuffle: bool =True,
                transformations = "default",
                 image_folder = "images",
                 label_folder = "groundtruth"):
    
    dataset = get_dataset(root_dir, transformations, image_folder, label_folder)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

def get_train_test_dataloaders(root_dir: str,
                               train_split: float = 0.8,
                               batch_size: int = 10,
                               shuffle: bool =True,
                                transformations = "default",
                                 image_folder = "images",
                                 label_folder = "groundtruth"):
    dataset = get_dataset(root_dir, transformations, image_folder, label_folder)
    
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*train_split), len(dataset)-int(len(dataset)*train_split)])
    
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle), torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    
    
    
"""
 # Usage example
 
from matplotlib import pyplot as plt

dataset = get_dataset("../data/training")
dataloader = get_dataloader("../data/training")
train_dataloader, test_dataloader = get_train_test_dataloaders("../data/training", train_split=0.8)

x, y = dataset.__getitem__(1)

plt.imshow(x.permute(1,2,0).numpy())

"""