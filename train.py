#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse

import torch

from models.model_builder import Summarizer

from utils import preprocess_data

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def train(args):
    device = "cpu"

    model = Summarizer(args, device, load_pretrained_bert=True)
    print(model)

    train_data, train_results, test_data, test_results = preprocess_data.get_data()

    ids = torch.tensor(train_data[0].ids)
    model(ids)  # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp_dir", default='temp')
    parser.add_argument("-encoder", default='classifier')
    parser.add_argument("-param_init", default=0.0)
    args = parser.parse_args()
    train(args)
