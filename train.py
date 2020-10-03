#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse

from models.model_builder import Summarizer

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def train(args):
    device = "cpu"

    model = Summarizer(args, device, load_pretrained_bert=True)
    print(model)
    model(['Hello', 'everyone', '.', 'Welcome', '.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-temp_dir", default='temp')
    parser.add_argument("-encoder", default='classifier')
    parser.add_argument("-param_init", default=0.0)
    args = parser.parse_args()
    # train(args)
