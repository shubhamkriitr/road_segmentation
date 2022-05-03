from argparse import ArgumentParser
from training.trainingutil import PIPELINE_NAME_TO_CLASS_MAP
import yaml

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = argparser.parse_args()

    config_data = None
    with open(args.config, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    assert config_data is not None, "Config file not found"

    pipeline_class = PIPELINE_NAME_TO_CLASS_MAP[config_data["pipeline_class"]]
    pipeline = pipeline_class(config=config_data)
    pipeline.prepare_experiment()
    pipeline.run_experiment()

