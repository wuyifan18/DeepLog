# parser.py

import os
import json
import argparse


def get_process_args():
    # Function to get command-line arguments

    # initialize argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--input-dir',
                        type=str,
                        help='Path to training data.')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Directory to save processed slogs.')
    parser.add_argument('--freq',
                        type=str,
                        default='1min',
                        help='Frequency to resample events together.')
    parser.add_argument('--tau',
                        type=float,
                        default=0.5,
                        help='Percentage to marge tokens')
    # parse args
    args = parser.parse_args()

    # create saved folder if not existing
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def get_model_args():
    # Function to get command-line arguments

    # initialize argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--config',
                        type=str,
                        help='Training config file.')
    parser.add_argument('--data',
                        type=str,
                        help='Path to training data.')
    parser.add_argument('--eval-data',
                        type=str,
                        default=None,
                        help='Path to evaluation data.')
    parser.add_argument('--vocab',
                        type=str,
                        default=None,
                        help='Path to vocab file.')
    # parse args
    args = parser.parse_args()

    # get training configs
    configs = parse_configs(args.config)

    # create saved folder if not existing
    if not os.path.exists(configs['saved']):
        os.makedirs(configs['saved'])

    return args, configs


def parse_configs(file):
    # Function to parse training params

    # read file expected to be in json
    file = file + '.json' if not file.endswith('.json') else file
    with open(file) as file:
        configs = json.load(file)
    return configs
