import argparse
import configparser
import subprocess
from openem_train import preprocess
from openem_train import detect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Top level training script.')
    parser.add_argument('config_file',
        help="Path to config file.")
    parser.add_argument('task',
        help="What task to do, one of: preprocess, detect.")
    args = parser.parse_args()

    # Read in the config file.
    config = configparser.ConfigParser()
    config.read(args.config_file)

    if args.task == 'preprocess':
        preprocess.extract_images(config)

    if args.task == 'detect':
        detect.train(config)

