# train.py

import os
import torch
from torch.utils.data import DataLoader

from parser import get_args
from a2l.dataset import LogDataset
from a2l.models import *


def main(args):
    # initialize datasets and convert to dataloaders
    train_dataset = LogDataset(args.data)
    train_dataset = DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               shuffle=args.shuffle,
                               pin_memory=True)

    if args.eval_data:
        eval_dataset = LogDataset(args.eval_data)
        eval_dataset = DataLoader(eval_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  pin_memory=True)

    pass


if __name__ == '__main__':
    # get argument arguments
    args = get_args()
    # the training pipeline with given args
    main(args)
