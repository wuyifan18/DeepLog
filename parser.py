# parser.py

import os
import argparse


def get_args():
    # Function to get command-line arguments

    # initialize argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--data',
                        type=str,
                        help='Path to training data.')
    parser.add_argument('--eval',
                        type=str,
                        default=None,
                        help='Path to evaluation data.')
    parser.add_argument('--iter',
                        type=int,
                        default=50,
                        help='Number of iterations')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Number of samples per batch.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.01,
                        help='Learning rate.')
    parser.add_argument('--lr-decay',
                        default=False,
                        action='store_true',
                        help='Flag to use the learning-rate decay.')
    parser.add_argument('--saved',
                        type=str,
                        default='./saved',
                        help='Directory to save trained models, params, and logs.')

    # create saved folder if not existing
    if not os.path.exists(args.saved):
        os.makedirs(args.saved)

    return parser.parse_args()
