#!/usr/bin/env python

"""
    detect.py
    
    Run `darknet` over stream of images
"""


import sys
import argparse
from time import time
from PIL import Image

from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox, format_image

defaults = {
    "name_path" : '/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data.bak/custom.names',
    "cfg_path" : '/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data.bak/yolo-custom.cfg',
    "weight_path" : '/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data.bak/backup/yolo-custom_10000.weights'    ,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name-path', type=str, default=defaults['name_path'])
    parser.add_argument('--cfg-path', type=str, default=defaults['cfg_path'])
    parser.add_argument('--weight-path', type=str, default=defaults['weight_path'])
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--draw', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    ObjectDetector.set_device(0)
    det = ObjectDetector(args.cfg_path, args.weight_path, args.thresh, args.nms, int(args.draw))
    
    class_names = open(args.name_path).read().splitlines()
    print class_names
    
    start = time()
    c_load_time, c_pred_time = 0, 0
    for i,im_name in enumerate(sys.stdin):
        py_all_time = time() - start
        py_load_time = py_all_time - c_load_time - c_pred_time
        print >> sys.stderr, "%d | pyall %f | pyload %f | cload %f | cpred %f" % (i, py_all_time, py_load_time, c_load_time, c_pred_time)
        
        try:
            im_name = im_name.strip()
            img = Image.open(im_name)
            img = format_image(img)
            rst, load_time, pred_time = det.detect_object(*img)
            # c_load_time += load_time
            # c_pred_time += pred_time
            
            # for bbox in rst:
            #     class_name = class_names[bbox.cls]
            #     res = [im_name, class_name, bbox.confidence, bbox.top, bbox.left, bbox.bottom, bbox.right]
            #     print '\t'.join(map(str, res))
            #     sys.stdout.flush()
        
        except KeyboardInterrupt:
            raise
        except:
            print >> sys.stderr, "Error @ %s" % im_name
