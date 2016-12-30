from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

import requests
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO
import cv2

def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))

if __name__ == '__main__':
    ObjectDetector.set_device(0)
    
    from PIL import Image
    names = open('../data/coco.names').read().splitlines()
    det = ObjectDetector('../cfg/yolo.cfg','../yolo.weights')
    
    img = Image.open("../data/dog.jpg")
    
    rst, run_time = det.detect_object(img)
    print 'got {} objects in {} seconds'.format(len(rst), run_time)

    for bbox in rst:
        print '{} {} {} {} {} {}'.format(names[bbox.cls], bbox.top, bbox.left, bbox.bottom, bbox.right, bbox.confidence)
