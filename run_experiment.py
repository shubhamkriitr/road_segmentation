from argparse import ArgumentParser
from training.trainingutil import PIPELINE_NAME_TO_CLASS_MAP, run_experiment
from utils.commonutil import read_config
import yaml

"""This is the main file called. It calls training.trainingutil.run_experiment"""

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = argparser.parse_args()
    run_experiment(read_config(args.config))
