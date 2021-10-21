# preprocess.py

import os
import re
import pandas as pd
from spellpy import spell

from parser import get_process_args

pd.options.mode.chained_assignment = None  # default='warn'


def transfer(df, event_id_map, freq='1min'):
    # Transfer series of logs into intervals of log series
    _custom_resampler = lambda x: list(x)

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df[['datetime', 'EventId']]
    df['EventId'] = df['EventId'].map(lambda e: event_id_map[e] if e in event_id_map else -1)
    deeplog_df = df.set_index('datetime').resample(freq).apply(_custom_resampler).reset_index()
    return deeplog_df


def generate(filename, df):
    # save logs
    with open(filename, 'w') as f:
        for event_id_list in df['EventId']:
            for event_id in event_id_list:
                f.write(str(event_id) + ' ')
            f.write('\n')

def process_hdfs(args, log_format):
    print('Parsing HDFS logs.')

    # initialize parser
    parser = spell.LogParser(
        indir=args.input_dir,
        outdir=args.output_dir,
        log_format=log_format,
        tau=args.tau)

    # parse all log files
    for file in os.listdir(args.input_dir):
        if file.endswith('.log'):
            parser.parse(file)

        # get event id map
        df = pd.read_csv(os.path.join(args.output_dir, 'HDFS.log_structured.csv'))
        event_id_map = dict()
        event_id_map['[UNK]'] = -1  # add unknown log key
        for i, event_id in enumerate(df['EventId'].unique(), 1):
            event_id_map[event_id] = i

        # transfer series of logs into intervals of log series
        for file in os.listdir(args.output_dir):
            if file.endswith('_structured.csv'):
                ds = pd.read_csv(os.path.join(args.output_dir, file))
                ds = transfer(ds, event_id_map, args.freq)
                generate(os.path.join(args.output_dir, re.sub('.csv', '.txt', file)), ds)

        # save vocab
        with open(os.path.join(args.output_dir, 'vocab.txt'), 'w') as file:
            file.write('\n'.join([str(v) + ' ' + str(k) for k,v in event_id_map.items()]))
    return None


def process_openstack(args, log_format):
    print('Parsing OpenStack logs.')

    # initialize parser
    parser = spell.LogParser(
        indir=args.input_dir,
        outdir=args.output_dir,
        log_format=log_format,
        tau=args.tau)

    # parse all log files
    for file in os.listdir(args.input_dir):
        if file.endswith('.log'):
            parser.parse(file)

    # get event id map
    df = pd.read_csv(os.path.join(args.output_dir, 'openstack_normal1.log_structured.csv'))
    event_id_map = dict()
    event_id_map['[UNK]'] = -1  # add unknown log key
    for i, event_id in enumerate(df['EventId'].unique(), 1):
        event_id_map[event_id] = i

    # transfer series of logs into intervals of log series
    for file in os.listdir(args.output_dir):
        if file.endswith('_structured.csv'):
            ds = pd.read_csv(os.path.join(args.output_dir, file))
            ds = transfer(ds, event_id_map, args.freq)
            generate(os.path.join(args.output_dir, re.sub('.csv', '.txt', file)), ds)

    # save vocab
    with open(os.path.join(args.output_dir, 'vocab.txt'), 'w') as file:
        file.write('\n'.join([str(v) + ' ' + str(k) for k,v in event_id_map.items()]))
    return None


def preproceess(args):
    # Currently, function supports either HDFS or OpenStack datasets

    # identify dataset
    if 'hdfs' in args.input_dir.lower():
        log_format = '<Date> <Time> <Pid> <Level> <Component> <Content> <EventId> <EventTemplate>'
        process_hdfs(args, log_format)
    elif 'openstack' in args.input_dir.lower():
        log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
        process_openstack(args, log_format)
    else:
        raise 'Dataset {} is not supproted.'.format(args.input_dir)

    return None


if __name__ == '__main__':
    # get processing args
    args = get_process_args()

    preproceess(args)
