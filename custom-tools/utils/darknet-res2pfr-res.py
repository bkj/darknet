#!/usr/bin/env python

"""
    darknet-res2pfr-res.py
    
    Converts output of `darknet` to format compatible with `pfr`
    
    - scales bounding boxes back to size of original image
    - reorders coordinates 
"""

import os
import sys
import json
import argparse
import shutil
from glob import glob

from PIL import Image

net_size = 416

if __name__ == "__main__":
    
    for (filename, class_name, score, ymin, xmin, ymax, xmax) in (line.strip().split('\t') for line in sys.stdin):
            
            score, xmin, ymin, xmax, ymax = map(float, (score, xmin, ymin, xmax, ymax))
            
            # !! Hack to scale bounding boxes
            im_size = Image.open(filename).size
            
            ymin = round((ymin / net_size) * im_size[1], 3)
            ymax = round((ymax / net_size) * im_size[1], 3)
            xmin = round((xmin / net_size) * im_size[0], 3)
            xmax = round((xmax / net_size) * im_size[0], 3)
            
            print '\t'.join(map(str, (filename, class_name, score, xmin, ymin, xmax, ymax)))



