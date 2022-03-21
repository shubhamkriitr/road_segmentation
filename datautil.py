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

    def __init__(self, rootdir, index_offset=0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rootdir = rootdir
        self.index_offset = index_offset
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
                       index_offset=0,
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
        index = index + self.index_offset
        input_image = self.load_image(  os.path.join(self.input_data_folder,
                                        self.index_to_filename(index)))
        input_image = input_image[:, :, 0:3] # Dropping fourth channel
        # as it has uninformative data (all values are 255)

        # torchvision requires image in [0, 1) range for float dtype
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
        input_image = input_image[:, :, 0:3] # Dropping fourth channel
        
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
            "group_probs" : [0.5, 0.125, 0.125, 0.125, 0.025, 0.1]
        }
        super().__init__(config=config) #self.config gets overridden by
        # `self.default_config` if `config` it is None

        self.transformation_group_list = [
            self.identity_transform,
            self.transfrom_group_a,
            self.transform_group_b,
            self.transform_group_c,
            self.transform_group_d,
            self.transform_group_e
        ]
        self.group_probs = self.config["group_probs"]
        assert len(self.transformation_group_list) == len(self.group_probs)
        self.probability_threshold = self.get_probability_thresolds(
            self.group_probs)

        # TODO: read spec from config and inject instance from outside
        self.rotation = SegRotationTransform()
        self.hflip = SegHorizontalFlip()
        self.vflip = SegVerticalFlip()
        self.brightness = SegAdjustBrightness()
        self.affine = SegAffine()
    
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
    
    def transform_group_e(self, input_, target):
        return self.affine(input_, target)

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
        self.brightness_factors = brightness_factors
        if self.brightness_factors is None:
            self.brightness_factors = [0.5, 0.8, 1.2, 1.5]

    def __call__(self, x, y):
        factor = random.choice(self.brightness_factors)
        return TF.adjust_brightness(x, brightness_factor=factor), y


class SegAffine:
    def __init__(self, angle=None, translate=None, scale=None,
                    shear=None) -> None:
        #no attrs
        self.angle = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120]
        self.translate = [[0, 1]]
        self.scale = [1.0, 1.2, 1.5, 0.8, 0.6]
        self.shear = [[15, 0], [0, 15], [15, 15], [25, 0], [0, 25], [25, 25]]

        if angle is not None:
            self.angle = angle
        if translate is not None:
            self.translate = translate
        if scale is not None:
            self.scale = scale
        if shear is not None:
            self.shear = shear


    def __call__(self, x, y):
        angle = random.choice(self.angle)
        translate = random.choice(self.translate)
        scale = random.choice(self.scale)
        shear = random.choice(self.shear)

        return TF.affine(x, angle=angle,
                translate=translate, scale=scale, shear=shear), \
                    TF.affine(y, angle=angle, translate=translate,
                     scale=scale, shear=shear)


def pad_zeros(num: int, size):
    num = str(num)
    return "0"*(size - len(num)) + num



def dump_transformed_images(loader, data_transformer_class, root_output_dir,
                             num_transforms=6):
    from matplotlib import pyplot as plt
    # num_transforms = no. of transformation groups (identity included)
    os.makedirs(root_output_dir, exist_ok=True)

    for idx, batch_data in enumerate(train_dataloader):
        
        if idx == num_transforms:
            break
        group_probs = [0]*num_transforms
        group_probs[idx] = 1.0

        data_aug = data_transformer_class(
            config={
                "group_probs" : group_probs
            }
        )
        x_, y_ = batch_data
        x_t, y_t = data_aug.transform(x_, y_)
        x, y = torch.transpose(x_t, 1, 2).transpose(2, 3), \
                     y_t.transpose(1, 2).transpose(2, 3)
        
        x_orig, y_orig = torch.transpose(x_, 1, 2).transpose(2, 3), \
                     y_.transpose(1, 2).transpose(2, 3)
        

        for j in range(x_.shape[0]):
            file_id = f"{pad_zeros(idx, 3)}_{pad_zeros(j, 3)}"
            orig_img_filename = os.path.join(root_output_dir, 
                                        file_id + "_orig_img.png")
            orig_map_filename = os.path.join(root_output_dir, 
                                            file_id + "_orig_map.png")
            aug_img_filename = os.path.join(root_output_dir, 
                                            file_id + "_aug_img.png")
            aug_map_filename = os.path.join(root_output_dir, 
                                            file_id + "_aug_map.png")
            
            plt.imsave(orig_img_filename, x_orig[j].numpy())
            plt.imsave(orig_map_filename, y_orig[j].numpy()[:, :, 0])
            plt.imsave(aug_img_filename, x[j].numpy())
            plt.imsave(aug_map_filename, y[j].numpy()[:, :, 0])



if __name__ == "__main__":

    ds = CILRoadSegmentationDataset(DEFAULT_TRAINING_DATASET_ROOTDIR)

    train_dataset = CILRoadSegmentationTrainingDataset(DEFAULT_TRAINING_DATASET_ROOTDIR)
    test_dataset = CILRoadSegmentationTestDataset(DEFAULT_TEST_DATASET_ROOTDIR)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    dump_transformed_images(train_dataloader, SegmentationTrainingDataTransformer, 
                            "test_transforms", 6)
    n1 = 2 
    n2 = 2

    from matplotlib import pyplot as plt

    data_aug = SegmentationTrainingDataTransformer(
        config={
            "group_probs" : [ 0.0, 0.0, 0., 0., 0., 1.0] #[0.5, 0.125, 0.125, 0.125, 0.025, 0.1]
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
       
        
        
