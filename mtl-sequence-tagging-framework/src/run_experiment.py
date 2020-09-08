"""
This module allows to run an experiment from a configuration template file.
"""

import argparse

from ConfigGenerator import ConfigGenerator
from use_network import train


def main():
    """
    Parse the CLI arguments and then run the experiment with different trials (i.e. hyper-parameter configurations).
    """
    parser = argparse.ArgumentParser(description="Running experiments with the MTL sequence tagging framework.")
    parser.add_argument("trials", help="The number of trials to perform", type=int)
    parser.add_argument("template", help="Path to the template file", type=str)
    parser.add_argument(
        "config_out", help="Directory where to output configuration files (may also be a temporary directory)"
    )

    args = parser.parse_args()

    config_generator = ConfigGenerator(args.template, args.config_out)

    for trial in range(args.trials):
        config_path = config_generator.generate()
        train(config_path)

if __name__ == "__main__":
    main()
