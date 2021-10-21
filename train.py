# train.py

import os
import torch
from torch.utils.data import DataLoader

from parser import get_model_args
from a2l.dataset import *
from a2l.models import *


def main(args, configs):
    # initialize datasets and convert to dataloaders
    train_dataset = LogDataset(args.data,
                               window_size=configs['window_size'],
                               vocab=args.vocab)
    train_dataset = DataLoader(train_dataset,
                               batch_size=configs['batch_size'],
                               shuffle=configs['shuffle'],
                               pin_memory=True)

    if args.eval_data:
        eval_dataset = LogDataset(args.eval_data,
                                  window_size=configs['window_size'],
                                  vocab=args.vocab)
        eval_dataset = DataLoader(eval_dataset,
                                  batch_size=configs['batch_size'],
                                  shuffle=configs['shuffle'],
                                  pin_memory=True)

    # initialize model
    model = LogTransformer(num_class=configs['num_class'],
                           encoder_hidden_size=configs['encoder_hidden_size'],
                           decoder_hidden_size=configs['decoder_hidden_size'],
                           num_layer=configs['num_layer'],
                           num_head=configs['num_head'],
                           dropout=configs['dropout'])
    print(model)

    # training pipeline
    pass


if __name__ == '__main__':
    # get argument arguments
    args, configs = get_model_args()
    # the training pipeline with given args
    main(args, configs)
