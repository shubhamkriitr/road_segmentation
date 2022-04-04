import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", "-i", default=None, type=str, help="Path to training data")
parser.add_argument("--output_path", "-o", default=None, type=str, help="Path to store split data")
parser.add_argument("--train_split", "-t", default=0.8, type=float, help="Percentage of images used to train. Default: 0.8")


# Sample execution: python split_data.py -i /Users/javi/Downloads/cil-road-segmentation-2022/training/ -o /Users/javi/Downloads/cil-road-segmentation-2022/split -t 0.8
if __name__ == "__main__":
    args = parser.parse_args()

    if args.input_path is None:
        raise Exception("You must provide a valid input path")

    if args.output_path is None:
        raise Exception("You must provide a valid output path")

    assert args.input_path != args.output_path, "Input and output paths must be different"

    files = [i for i in os.listdir(os.path.join(args.input_path, "images")) if i.endswith(".png")]

    random.seed(4)
    random.shuffle(files)

    train_files = files[:int(len(files) * 0.8)]
    test_files = files[int(len(files) * 0.8):]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Make training and test folders
    out_train_folder = os.path.join(args.output_path, "train")
    out_test_folder = os.path.join(args.output_path, "test")

    # Remove existing content to avoid problems
    if os.path.exists(out_train_folder):
        os.rmdir(out_train_folder)
    os.makedirs(os.path.join(out_train_folder, "images"))
    os.makedirs(os.path.join(out_train_folder, "groundtruth"))

    if os.path.exists(out_test_folder):
        os.rmdir(out_test_folder)
    os.makedirs(os.path.join(out_test_folder, "images"))
    os.makedirs(os.path.join(out_test_folder, "groundtruth"))

    # Create content
    for file in train_files:
        shutil.copy(os.path.join(args.input_path, "images", file), os.path.join(out_train_folder, "images"))
        shutil.copy(os.path.join(args.input_path, "groundtruth", file), os.path.join(out_train_folder, "groundtruth"))

    # Create content
    for file in test_files:
        shutil.copy(os.path.join(args.input_path, "images", file), os.path.join(out_test_folder, "images"))
        shutil.copy(os.path.join(args.input_path, "images", file), os.path.join(out_test_folder, "images"))
