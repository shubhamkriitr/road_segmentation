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

# Set random seed
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)


# Using prefix `Seg` (for segmentation) to the classes below to distinguish
# them from torchvision's  classes
class SegRotationTransform:
    def __init__(self, angles=None):
        self.angles = angles
        if self.angles is None:
            self.angles = [90, 180, -90, -180]

    def __call__(self, x, y):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle), TF.rotate(y, angle)


class SegHorizontalFlip:
    def __init__(self, *args, **kwargs) -> None:
        # no attrs
        pass

    def __call__(self, x, y):
        return TF.hflip(x), TF.hflip(y)


class SegVerticalFlip:
    def __init__(self, *args, **kwargs) -> None:
        # no attrs
        pass

    def __call__(self, x, y):
        return TF.vflip(x), TF.vflip(y)


class IdentityTransform:
    def __init__(self, *args, **kwargs) -> None:
        # no attrs
        pass

    def __call__(self, x, y):
        return x, y


class SegAdjustBrightness:
    def __init__(self, brightness_factors=None) -> None:
        # no attrs
        self.brightness_factors = brightness_factors
        if self.brightness_factors is None:
            self.brightness_factors = [1, 1.1, 1.2, 0.9]

    def __call__(self, x, y):
        factor = random.choice(self.brightness_factors)
        return TF.adjust_brightness(x, brightness_factor=factor), y


class SegAdjustContrast:
    def __init__(self, constrast_factors=None) -> None:
        # no attrs
        self.constrast_factors = constrast_factors
        if self.constrast_factors is None:
            self.constrast_factors = [0.8, 0.9, 1.2, 1.5]

    def __call__(self, x, y):
        factor = random.choice(self.constrast_factors)
        return TF.adjust_contrast(x, factor), y


class SegNormalize:
    def __init__(self, mean=(0.5123, 0.5233, 0.5206), std=(0.2425, 0.2210, 0.2117)) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x, y):
        return TF.normalize(x, self.mean, self.std), y


class SegGaussianBlur:
    def __call__(self, x, y):
        return TF.gaussian_blur(x, kernel_size=(5, 9), sigma=(0.1, 5)), y


# These will be performed as a first step
default_base_transformations = {IdentityTransform(): 0.25,
                                SegHorizontalFlip(): 0.25,
                                SegVerticalFlip(): 0.25,
                                SegRotationTransform(): 0.25}

# Then these will be applied on top of the previous ones
default_additional_transformations = {IdentityTransform(): 0.30,
                                      SegRotationTransform(): 0.35,
                                      SegAdjustBrightness(): 0.15,
                                      SegGaussianBlur(): 0.05,
                                      SegAdjustContrast(): 0.15}


class CILRoadSegmentationDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 image_folder="images",
                 label_folder="groundtruth",  # Set to none if no supervision
                 base_transformations=default_base_transformations,  # Dictionary where key is class to be applied, value is the probability
                 additional_transformations=default_additional_transformations,
                 normalize=True):

        self.root_dir = root_dir
        self.num_samples = 0
        self.image_names = []

        # Define paths for both images and labels given the root directory
        self.images_path = os.path.join(self.root_dir, image_folder)
        self.labels_path = os.path.join(self.root_dir, label_folder) if label_folder else None

        # Store transformations
        self.base_transformations = base_transformations
        self.additional_transformations = additional_transformations
        self.normalize = normalize
        if self.normalize:
            self.normalize_transform = SegNormalize()

        # Store name of images
        self._inspect_rootdir()

    def _inspect_rootdir(self):
        is_img = lambda x: '.png' in x
        self.image_names = [f for f in os.listdir(self.images_path) if is_img(f)]
        self.num_samples = len(self.image_names)

        if self.labels_path:
            assert self.num_samples == sum(is_img(filename) for filename in os.listdir(self.labels_path)), "Number of images does not match number of labels"

        if self.base_transformations:
            assert np.array(list(self.base_transformations.values())).sum() == 1, "Probabilities of transformations should up to 1"

        if self.additional_transformations:
            assert np.array(list(self.additional_transformations.values())).sum() == 1, "Probabilities of transformations should up to 1"

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
        input_image = np.clip(input_image / 255, a_min=0, a_max=1 - 1e-7)

        if self.labels_path:
            groundtruth = self.load_image(os.path.join(self.labels_path, self.image_names[index]))

            # NOTE: creating probability map from the ground truth image
            groundtruth = torch.tensor(groundtruth) / 255
            # groundtruth = torch.stack([groundtruth, 1 - groundtruth], dim=0)
            groundtruth = groundtruth.unsqueeze(0)
        else:
            groundtruth = None

        # To tensors
        input_image = self.to_tensor(input_image).transpose(1, 2).transpose(0, 1)

        # remove unnecessary dim
        input_image = input_image[:3]

        if self.base_transformations and self.labels_path:
            # Get transform according to probabilities
            transform = np.random.choice(list(self.base_transformations.keys()), size=1, p=list(self.base_transformations.values()))[0]
            input_image, groundtruth = transform(input_image, groundtruth)

            if self.additional_transformations:
                transform = np.random.choice(list(self.additional_transformations.keys()), size=1, p=list(self.additional_transformations.values()))[0]
                input_image, groundtruth = transform(input_image, groundtruth)

        if self.normalize:
            input_image, groundtruth = self.normalize_transform(input_image, groundtruth)

        return input_image, groundtruth


def get_dataset(root_dir: str,
                base_transformations="default",
                additional_transformations="default",
                normalize=True,
                image_folder="images",
                label_folder="groundtruth"):
    base_transformations = default_base_transformations if base_transformations == "default" else base_transformations if type(base_transformations) is dict else None
    additional_transformations = default_additional_transformations if additional_transformations == "default" else additional_transformations if type(additional_transformations) is dict else None

    return CILRoadSegmentationDataset(root_dir=root_dir,
                                      base_transformations=base_transformations,
                                      additional_transformations=additional_transformations,
                                      normalize=normalize,
                                      image_folder=image_folder,
                                      label_folder=label_folder)


def get_train_test_dataloaders(root_dir: str,
                               batch_size: int = 10,
                               shuffle: bool = True,
                               normalize: bool = True,
                               base_transformations="default",
                               additional_transformations="default",
                               image_folder="images",
                               label_folder="groundtruth"):
    assert "train" in os.listdir(root_dir) and "test" in os.listdir(root_dir), "You must provide the path to the split/ folder"
    train_dataset = get_dataset(os.path.join(root_dir, 'train'), base_transformations, additional_transformations, normalize, image_folder, label_folder)
    test_dataset = get_dataset(os.path.join(root_dir, 'test'), None, None, normalize, image_folder, label_folder)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


class VanillaDataLoaderUtil(object):
    def __init__(self, config=None) -> None:
        self.config = config # currently not in use (adding for extension)#TODO 

    def get_data_loaders(self, root_dir: str,
                               batch_size: int = 10,
                               shuffle: bool = True,
                               normalize: bool = True):
        """
        Returns tuple of (train_loader, val_loader, test_loader)
        """
        train_loader, val_loader = get_train_test_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            normalize=normalize
        )
        return train_loader, val_loader, None # no test loader
        
        
DATALOADER_UTIL_CLASS_MAP = {
    "VanillaDataLoaderUtil": VanillaDataLoaderUtil
}

class DataLoaderUtilFactory:
    def __init__(self, config=None) -> None:
        self.config = config
        self.resource_map = DATALOADER_UTIL_CLASS_MAP
    
    def get(self, dataloader_util_class_name, config=None):
        # currently not using config #TODO
        loader_util_class = self.resource_map[dataloader_util_class_name]
        try:
            return loader_util_class(config=config)
        except TypeError as err:
            if config is not None:
                raise err
        return loader_util_class()
        
"""
>>>  # Usage example
>>>  
>>> from matplotlib import pyplot as plt
>>> 
>>> dataset = get_dataset("../data/training")
>>> dataloader = get_dataloader("../data/training")
>>> train_dataloader, test_dataloader = get_train_test_dataloaders("../data/training", train_split=0.8)
>>> 
>>> x, y = dataset.__getitem__(1)
>>> 
>>> plt.imshow(x.permute(1,2,0).numpy())

"""
