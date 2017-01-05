"""
    detect.py
    
    Run `darknet` over stream of images
"""


import sys
import argparse
from time import time
from PIL import Image

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


if __name__ == "__main__":
    args = parse_args()
    
    ObjectDetector.set_device(0)
    det = ObjectDetector(args.cfg_path, args.weight_path, args.thresh, args.nms, int(args.draw))
    
    class_names = open(args.name_path).read().splitlines()
    print class_names
    
    start = time()
    for i,im_name in enumerate(sys.stdin):
        print >> sys.stderr, "Processed %d images in %f seconds" % (i, time() - start)
        
        try:
            im_name = im_name.strip()
            img = Image.open(im_name)
            rst, run_time = det.detect_object(*det.format_image(img))
            
            for bbox in rst:
                class_name = class_names[bbox.cls]
                res = [im_name, class_name, bbox.confidence, bbox.top, bbox.left, bbox.bottom, bbox.right]
                print '\t'.join(map(str, res))
        except KeyboardInterrupt:
            raise
        except:
            print >> sys.stderr, "Error @ %s" % im_name
