"""
Main module that can be used to interact with the network via the CLI.
The module can be invoked in three modes: train, eval, and predict (NOTE: predict is not implemented yet).

Examples:
```
    # Train
    python main.py train my_configuration.yaml

    # Evaluate
    python main.py eval path_to_my_saved_model my_configuration.yaml

    # Predict
    python main.py predict path_to_my_saved_model my_configuration.yaml
```
"""
import sys

from use_network import train, evaluate, predict

if __name__ == "__main__":
    MODE_TRAIN = "train"
    MODE_EVAL = "eval"
    MODE_PREDICT = "predict"
    VALID_MODES = [MODE_TRAIN, MODE_EVAL, MODE_PREDICT]

    if sys.argv is None or len(sys.argv) < 2:
        sys.stderr.write("Please specify a mode and at least one configuration file.")
        sys.stderr.write("For example: python main.py train my_config.yaml")
        sys.exit(1)

    mode = sys.argv[1]

    if mode not in VALID_MODES:
        sys.stderr.write("Please select a mode which is one of %s" % VALID_MODES)
        sys.exit(1)

    print("Running in mode '%s'" % mode)

    if mode == MODE_TRAIN:
        configuration_files = sys.argv[2:]
        print("Called train for %d different configuration files" % len(configuration_files))
        print("Files: %s" % ", ".join(configuration_files))

        for configuration_file in configuration_files:
            train(configuration_file, False)
    elif mode == MODE_EVAL:
        model_path = sys.argv[2]
        configuration_files = sys.argv[3:]
        print("Called evaluate for %d different configuration files" % len(configuration_files))
        print("Files: %s" % ", ".join(configuration_files))
        print("Using model at %s" % model_path)

        for configuration_file in configuration_files:
            evaluate(configuration_file, model_path)
    elif mode == MODE_PREDICT:
        model_path = sys.argv[2]
        configuration_files = sys.argv[3:]
        print("Called predict for %d different configuration files" % len(configuration_files))
        print("Files: %s" % ", ".join(configuration_files))
        print("Using model at %s" % model_path)

        for configuration_file in configuration_files:
            predict(configuration_file, model_path)
