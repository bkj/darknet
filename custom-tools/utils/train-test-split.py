#!/usr/bin/env python

"""
    train-test-split.py
    
    - Split dataset into train/test sets

    !! Right now we're doing this naively.  But we could add
        - stratified sampling
        - deduplication
"""

import os
import argparse
from glob import glob
import numpy as np
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--train-size', type=float, default=0.75)
    parser.add_argument('--random-seed', type=int, default=123)
    parser.add_argument('--keep-extensions', action='store_true')
    return parser.parse_args()

def _drop_extensions_gen(ann_in):
    for f in glob(os.path.join(ann_in, '*xml')):
        yield os.path.basename(f).split('.')[0] + '\n'

def _keep_extensions_gen(ann_in):
    for f in glob(os.path.join(ann_in, '*xml')):
        tree = ET.parse(open(f))
        filename = tree.find('filename').text
        if ('jpg' in filename) or ('JPEG' in filename) or ('jpeg' in filename):
            yield filename + '\n'

if __name__ == "__main__":
    args = parse_args()
    
    np.random.seed(args.random_seed)
    
    ann_in = os.path.join(args.indir, 'annotations')
    
    train = open(os.path.join(args.indir, 'image_sets', 'trainval.txt'), 'w')
    test = open(os.path.join(args.indir, 'image_sets', 'test.txt'), 'w')
    
    if not args.keep_extensions:
        gen = _drop_extensions_gen(ann_in)
    else:
        gen = _keep_extensions_gen(ann_in)
    
    for line in gen:        
        if np.random.uniform() < args.train_size:
            train.write(line)
        else:
            test.write(line)
    
    train.close()
    test.close()