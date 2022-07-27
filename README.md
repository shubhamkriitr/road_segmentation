

# Replication using provided setup scripts

- We have tested our code in an environment with

- Python 3.8.13
- OS
```
        LSB Version:	:core-4.1-amd64:core-4.1-noarch
        Distributor ID:	CentOS
        Description:	CentOS Linux release 7.9.2009 (Core)
        Release:	7.9.2009
        Codename:	Core
``` 
- CUDA
```
        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2020 NVIDIA Corporation
        Built on Mon_Oct_12_20:09:46_PDT_2020
        Cuda compilation tools, release 11.1, V11.1.105
        Build cuda_11.1.TC455_06.29190527_0
```

To replicate our experiments, it should be possible to put the dataset zip (the official one provided with the project - https://polybox.ethz.ch/index.php/s/AGkDmbC8IfmtBkr/download?path=%2F&files=cil-road-segmentation-2022.zip) in the repository, call `source ./setup.sh` to prepare the environment and finally `source ./replicate.sh` to run the experiment. However please note that these scripts have not undergone extensive testing, so read the rest of the README and if needed, follow the appropriate steps manually. Also, both of these scripts need internet access. 

# Manual replication
The configuration files of the final experiment are located in `experiment_configs/final`. The main config to be used is `config.yaml`. The experiments can be run by `python3 run_experiment.py --config config.yaml`. _e.g._ `python3 run_experiment.py --config experiment_configs/exp_04a2_resnet50_gdl.yaml`, _for experimenting with Resnet50 with Generlaized Dice Loss._

A checkpoint called `globe.ckpt` is needed. This is the checkpoint after training ConvNext on DeepGlobe data and can be downloaded from https://polybox.ethz.ch/index.php/s/1fAbWrYUuf3oLWP.
Furthermore, to run any experiments, the path to the data needs to be specified in the config under the name `data_root_dir`, which is set to `data/` by default. This directory is expected to contain the following directory structure:
- split
    - train
        - groundtruth
        - images
    - test
        - groundtruth
        - images
- test
    - images

The model is trained on the images from `split/train`, `split/test` is used as a validation set. We used a validation set consisting of the following images:
- satimage_107.png
- satimage_126.png
- satimage_129.png
- satimage_131.png
- satimage_133.png
- satimage_138.png
- satimage_139.png
- satimage_21.png
- satimage_22.png
- satimage_29.png
- satimage_30.png
- satimage_32.png
- satimage_35.png
- satimage_37.png
- satimage_45.png
- satimage_54.png
- satimage_57.png
- satimage_60.png
- satimage_63.png
- satimage_65.png
- satimage_69.png
- satimage_6.png
- satimage_70.png
- satimage_83.png
- satimage_84.png
- satimage_85.png
- satimage_89.png
- satimage_94.png
- satimage_95.png

For the final experiment, we included the validation set in the training data (by copying the aforementioned images to `split/train`). We left copies of these images in `split/test`, consequently the validation numbers were generated but unreliable. The behavior of the pipeline in case the images were deleted from `split/test` is unknown to us.

The required packages are listed in `requirements.txt` and can be installed using `pip3 install -r requirements.txt`.

# Other notes
We used python3 3.8.13+ during development.

The configuration of the experiment with regular (non-generalized) Dice loss (also reaching 93.3%) can be found in `experiment_configs/old_dice_final`. Note that when ensembling is requested in the config, the configs to be ensembled are looked for on the path relative to the current working directory (not the original config directory). The same holds for the path to the data directory. Consequently, the working directory matters and for some configs it might be required to move some files. We took steps to ensure no file movement is needed for replicating the `final` and `old_dice_final` experiments provided the working directory is the root of the repository.

Please note that the Dice loss which is referred from the report is DiceLossV2 from this repository and if ensembles are used, the validation scores are also generated but unreliable.

Moreover, on the first run, ConvNext weights need to be downloaded, so please make sure the script has internet access, otherwise you may encounter problems even if you provide internet access later. Also, a number of checkpoints is saved, which takes up quite a bit of storage under the `runs` directory. If needed, it should help to change the save frequency in the config.
