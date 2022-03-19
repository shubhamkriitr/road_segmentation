from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import torchvision.transforms.functional as TF
import torchvision
import torch
import logging
import PIL
import numpy as np
import cv2
import random

cv2.threshold

logger = logging.getLogger(name=__name__)

INPUT_FOLDER_NAME = "images"
TRUE_OUTPUT_FOLDER_NAME = "groundtruth"
DEFAULT_TRAINING_DATASET_ROOTDIR \
    = "resources/dataset/cil-road-segmentation-2022/training"
DEFAULT_TEST_DATASET_ROOTDIR \
    = "resources/dataset/cil-road-segmentation-2022/test"
IMG_EXT = ".png"
IMG_PREFIX = "satimage_"

# TODO: better to take this from run config sigleton
TENSOR_FLOAT_DTYPE = torch.float32

# test files are numbered from 144
TEST_DATASET_INDEX_OFFSET = 144


class CILRoadSegmentationDataset(Dataset):

    def __init__(self, rootdir, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rootdir = rootdir
        self.num_samples = 0
        self.input_data_folder = None
        self.groundtruth_output_data_folder = None

        self._inspect_rootdir()

    def index_to_filename(self, index):
        return f"{IMG_PREFIX}{index}{IMG_EXT}"

    def _inspect_rootdir(self):
        content_list = os.listdir(self.rootdir)
        
        if TRUE_OUTPUT_FOLDER_NAME in content_list:
            self.groundtruth_output_data_folder = Path(
                self.rootdir, TRUE_OUTPUT_FOLDER_NAME
            )
        if INPUT_FOLDER_NAME in content_list:
            self.input_data_folder = Path(
                self.rootdir, INPUT_FOLDER_NAME
            )
        
        if self.input_data_folder is None:
            raise AssertionError()
        
        
        #check num samples
        filelist = os.listdir(self.input_data_folder)
        is_img = lambda x: IMG_EXT in x
        self.num_samples = sum( is_img(filename) for filename in filelist)

        if self.groundtruth_output_data_folder is not None:
            filelist = os.listdir(self.groundtruth_output_data_folder)
            n = sum( is_img(filename) for filename in filelist)
            if n != self.num_samples:
                raise AssertionError("Number of input images must be the same "
                    "as the number of outputs")


    def load_image(self, image_path):
        # N.B.: img_array is of unit8 dtype with pixel value range [0-255]
        # normalize it later in the pipeline or store normalized version
        img_array = np.asarray(PIL.Image.open(image_path))
        return img_array

    def to_tensor(self, img_array):
        return torch.tensor(data=img_array, dtype=TENSOR_FLOAT_DTYPE)
        

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int):
        raise NotImplementedError()


class CILRoadSegmentationTrainingDataset(CILRoadSegmentationDataset):
    def __init__(self, rootdir=DEFAULT_TRAINING_DATASET_ROOTDIR,
                 *args, **kwargs) -> None:
        super().__init__(rootdir, *args, **kwargs)
        if self.groundtruth_output_data_folder is None:
            raise AssertionError("output folder missing")

    def _inspect_rootdir(self):
        super()._inspect_rootdir()

    def __len__(self) -> int:
        # N.B Warning!: while splitting it further using torch's API
        # this value will become inconsitent 
        return self.num_samples

    def __getitem__(self, index: int):
        input_image = self.load_image(  os.path.join(self.input_data_folder,
                                        self.index_to_filename(index)))
        # torchvision requires image in [0, 1) range for float dtype
        input_image = input_image[:, :, 0:3] # Dropping fourth channel
        # as it has uninformative data (all values are 255)
        input_image = np.clip(input_image/255, a_min=0, a_max= 1 - 1e-7)

        output_map = self.load_image(
            os.path.join(self.groundtruth_output_data_folder,
                    self.index_to_filename(index)))

        # NOTE: creating probability map from the ground truth image
        output_map = output_map//255

        return self.to_tensor(input_image).transpose(1, 2).transpose(0, 1),\
         self.to_tensor(output_map).unsqueeze(0)

class CILRoadSegmentationTestDataset(CILRoadSegmentationDataset):
    def __init__(self, rootdir=DEFAULT_TEST_DATASET_ROOTDIR,
                 *args, **kwargs) -> None:
        super().__init__(rootdir, *args, **kwargs)

    def _inspect_rootdir(self):
        super()._inspect_rootdir()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        index = index + TEST_DATASET_INDEX_OFFSET
        input_image = self.load_image(Path(self.input_data_folder,
                                            self.index_to_filename(index)))
        # torchvision requires image in [0, 1) range for float dtype
        input_image = np.clip(input_image/255, a_min=0, a_max= 1 - 1e-7)
        return self.to_tensor(input_image).transpose(1, 2).transpose(0, 1)


# NOTE: https://pytorch.org/vision/stable/transforms.html
# https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py

class BaseSegmentationDataTransformer:

    def __init__(self, config=None) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = self.default_config
    
    def transform(*args, **kwargs):
        raise NotImplementedError()


class SegmentationTrainingDataTransformer(BaseSegmentationDataTransformer):

    def __init__(self, config=None) -> None:
        self.default_config = {
            "group_probs" : [0.5, 0.125, 0.125, 0.125, 0.125]
        }
        super().__init__(config=config) #self.config gets overridden by
        # `self.default_config` if `config` it is None

        self.transformation_group_list = [
            self.identity_transform,
            self.transfrom_group_a,
            self.transform_group_b,
            self.transform_group_c,
            self.transform_group_d
        ]
        self.group_probs = self.config["group_probs"]
        assert len(self.transformation_group_list) == len(self.group_probs)
        self.probability_threshold = self.get_probability_thresolds(
            self.group_probs)

        self.rotation = SegRotationTransform()
        self.hflip = SegHorizontalFlip()
        self.vflip = SegVerticalFlip()
        self.brightness = SegAdjustBrightness()
    
    def identity_transform(self, input_, target):
        return input_, target

    def transform(self, input_, target):
        # Note some transfroms should only change the input (X) not Y
        # while others it should mirror in Y as well
        

        p = np.random.random()
        transformed_input, transformed_target = input_, target
        for idx, p_threshold in enumerate(self.probability_threshold):
            if p <= p_threshold:
                transformed_input, transformed_target \
                    = self.transformation_group_list[idx](input_, target)
                return transformed_input, transformed_target


    
    def get_probability_thresolds(self, probability_list):
        prob_threshold = []
        running_sum = 0
        for p in probability_list:
            running_sum += p
            prob_threshold.append(running_sum)
        
        assert prob_threshold[-1] == 1.0
        return prob_threshold

    
    def transfrom_group_a(self, input_, target):
        #probability of components of each group must sum to 1
        return self.rotation(input_, target)

    def transform_group_b(self, input_, target):
        return self.hflip(input_, target)

    def transform_group_c(self, input_, target):
        return self.vflip(input_, target)
    
    def transform_group_d(self, input_, target):
        return self.brightness(input_, target)

    def register_transformation(self, transformation):
        # TODO: to be implemented if needed
        raise NotImplementedError()

# Transforms

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

class SegAdjustBrightness:
    def __init__(self, brightness_factors=None) -> None:
        #no attrs
        self.brightness_factors = brightness_factors
        if self.brightness_factors is None:
            self.brightness_factors = [0.5, 0.8, 1.2, 1.5]

    def __call__(self, x, y):
        factor = random.choice(self.brightness_factors)
        return TF.adjust_brightness(x, brightness_factor=factor), y

        



if __name__ == "__main__":

    ds = CILRoadSegmentationDataset(DEFAULT_TRAINING_DATASET_ROOTDIR)

    train_dataset = CILRoadSegmentationTrainingDataset(DEFAULT_TRAINING_DATASET_ROOTDIR)
    test_dataset = CILRoadSegmentationTestDataset(DEFAULT_TEST_DATASET_ROOTDIR)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    n1 = 2 
    n2 = 2

    from matplotlib import pyplot as plt

    data_aug = SegmentationTrainingDataTransformer(
        config={
            "group_probs" : [ 0.0, 1.0, 0., 0., 0.] #[0.5, 0.125, 0.125, 0.125, 0.125]
        }
    )

    for idx, batch_data in enumerate(train_dataloader):
        if idx == n1:
            break
        x_, y_ = batch_data
        x_t, y_t = data_aug.transform(x_, y_)

        x, y = torch.transpose(x_t, 1, 2).transpose(2, 3), y_t.transpose(1, 2).transpose(2, 3)
        # 
        plt.imshow(x[0].numpy())
        plt.show()
        if y is not None:
            plt.imshow(y[0].numpy())
            plt.show()

        plt.imshow(x[-1].numpy())
        plt.show()
        if y is not None:
            plt.imshow(y[-1].numpy())
            plt.show()

    for idx, batch_data in enumerate(test_dataloader):
        if idx == n2:
            break
        x = batch_data
        plt.imshow(x[0].numpy())
        plt.show()
        

        plt.imshow(x[-1].numpy())
        plt.show()
       
        
        
