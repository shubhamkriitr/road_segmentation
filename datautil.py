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
        input_image = np.clip(input_image/255, a_min=0, a_max= 1 - 1e-7)

        output_map = self.load_image(
            os.path.join(self.groundtruth_output_data_folder,
                    self.index_to_filename(index)))

        # NOTE: creating probability map from the ground truth image
        output_map = output_map//255

        return self.to_tensor(input_image), self.to_tensor(output_map)

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
        return self.to_tensor(input_image)



if __name__ == "__main__":

    ds = CILRoadSegmentationDataset(DEFAULT_TRAINING_DATASET_ROOTDIR)

    train_dataset = CILRoadSegmentationTrainingDataset(DEFAULT_TRAINING_DATASET_ROOTDIR)
    test_dataset = CILRoadSegmentationTestDataset(DEFAULT_TEST_DATASET_ROOTDIR)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    n1 = 2 
    n2 = 2

    from matplotlib import pyplot as plt

    for idx, batch_data in enumerate(train_dataloader):
        if idx == n1:
            break
        x, y = batch_data
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
       
        
        
