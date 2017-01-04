#!/usr/bin/env python

"""
    res2sloth-darknet.py
    
    Converts output of `darknet detect` back to `sloth` JSON so we can iterate on the model
"""

import os
import sys
import json
import argparse
import shutil
from glob import glob
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--class-name', type=str, required=True)
    return parser.parse_args()

def reduce_anns(anns):
    anns = anns.items()
    anns = sorted(anns, key=lambda x: -max([y['score'] for y in x[1]]))
    for ann in anns:
        for a in ann[1]:
            del a['score']
    
    return [{"class" : "image", "filename" : k, "annotations" : v} for k,v in anns]

def make_lookup():
    fs = glob('./test-data/three/*/*')
    return dict([(os.path.basename(f).split('.')[0], f) for f in fs])


if __name__ == "__main__":
    
    args = parse_args()
    
    if not os.path.exists(args.output):
        print >> sys.stderr, 'making output directory: %s' % args.output
        os.makedirs(args.output)
    
    lookup = make_lookup()
    
    anns = defaultdict(list)
    for i, (filename, score, xmin, ymin, xmax, ymax) in enumerate((line.strip().split(' ') for line in sys.stdin)):
        
        basename = os.path.basename(filename)
        try:
            if not os.path.exists(os.path.join(args.output, basename)):
                shutil.copy(lookup[filename], os.path.join(args.output, basename))
                print >> sys.stderr, 'Moved %s' % filename
        except KeyboardInterrupt:
            raise
        except:
            print >> sys.stderr, 'Cannot move %s' % filename
        
        score, xmin, ymin, xmax, ymax = map(float, (score, xmin, ymin, xmax, ymax))
        anns[basename].append({
            "score": score,
            "class": args.class_name,
            "height": ymax - ymin,
            "width": xmax - xmin,
            "x": xmin,
            "y": ymin
        })
    
    json.dump(reduce_anns(anns), open(os.path.join(args.output, 'anns.json'), 'w'), indent=2)


