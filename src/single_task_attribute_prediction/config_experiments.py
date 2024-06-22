import argparse
import yaml


def add_args(parser):  # common arguments for all CLIs
    parser.add_argument(
        "--model_path", type=str, required=False, help="Path to the model file"
    )
    parser.add_argument("--config", type=str, required=True, help="config file path")


def parse_args():
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser.parse_args()


def get_config():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


config = get_config()
