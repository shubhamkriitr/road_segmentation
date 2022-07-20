The configuration files of the final experiment are located in `experiment_configs/final`. The main config to be used is `config.yaml`. The experiments can be run by `python3 run_experiment.py --config config.yaml`.
To replicate the final experiments, a checkpoint called `globe.ckpt` is needed. This is the checkpoint after training ConvNext on DeepGlobe data and can be downloaded from https://polybox.ethz.ch/index.php/s/1fAbWrYUuf3oLWP.
Furthermore, to run any experiments, the path to the data needs to be specified in the config under the name `data_root_dir`, which is set to `data/` by default. This directory is expected to contain the following directory structure:
- eval_split
    - train
        - groundtruth
        - images
    - val
        - groundtruth
        - images

The model is trained on the images from `eval_split/train` and `eval_split/val` is used as a validation set. We used a validation set consisting of the following images:
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

For the final experiment, we included the validation set in the training data (by copying the aforementioned images to `eval_split/train`). We left copies of these images in `eval_split/val`, consequently the validation numbers were generated but unreliable. The behavior of the pipeline in case the images were deleted from `eval_split/val` is unknown to us.

The configuration of the experiment with regular (non-generalized) Dice loss (also reaching 93.3%) can be found in `experiment_configs/old_dice_final`.

Please note that the Dice loss which is referred from the report is DiceLossV2 from this repository and if ensembles are used, the validation scores are also generated but unreliable.
