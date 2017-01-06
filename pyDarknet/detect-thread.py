"""
    detect-thread.py
    
    Run `darknet` over stream of images
"""

import os
import sys
import numpy as np
from time import sleep
import argparse
from time import time
from PIL import Image

from multiprocessing import Process, Queue

from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

defaults = {
    "name_path" : '/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data/custom.names',
    "cfg_path" : '/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data/yolo-custom.cfg',
    "weight_path" : '/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data/backup/yolo-custom_10000.weights'    ,
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

def prep_images(in_, out_, det):
    while True:
        im_name = in_.get()
        try:
            pil_image = Image.open(im_name)
            out_.put((im_name, det.format_image(pil_image)))
        except KeyboardInterrupt:
            raise
        except:
            print >> sys.stderr, "Error @ %s" % im_name

def read_stdin(gen, out_):
    for line in gen:
        line = line.strip()
        out_.put(line)

if __name__ == "__main__":
    args = parse_args()
    
    ObjectDetector.set_device(0)
    det = ObjectDetector(args.cfg_path, args.weight_path, args.thresh, args.nms, int(args.draw))
    
    class_names = open(args.name_path).read().splitlines()
    print class_names
    
    # Thread to read from std
    filenames = Queue()
    newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
    stdin_reader = Process(target=read_stdin, args=(newstdin, filenames))
    stdin_reader.start()
    
    # Thread to load images    
    processed_images = Queue()
    image_processors = [Process(target=prep_images, args=(filenames, processed_images, det)) for _ in range(10)]
    for image_processor in image_processors:
        image_processor.start()
    
    i = 0
    start = time()
    c_load_time, c_pred_time = 0, 0
    while True:
        py_all_time = time() - start
        py_load_time = py_all_time - c_load_time - c_pred_time
        print >> sys.stderr, "%d | pyall %f | pyload %f | cload %f | cpred %f" % (i, py_all_time, py_load_time, c_load_time, c_pred_time)
        i += 1
        
        im_name, img = processed_images.get()
        rst, load_time, pred_time = det.detect_object(*img)
        c_load_time += load_time
        c_pred_time += pred_time
        
        for bbox in rst:
            class_name = class_names[bbox.cls]
            res = [im_name, class_name, bbox.confidence, bbox.top, bbox.left, bbox.bottom, bbox.right]
            print '\t'.join(map(str, res))
            sys.stdout.flush()