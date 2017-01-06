from libpydarknet import DarknetObjectDetector

import sys
import numpy as np
from PIL import Image

class DetBBox(object):

    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls

class Darknet_ObjectDetector():

    def __init__(self, spec, weight, thresh=0.5, nms=0.4, draw=0, py_resize=False, net_size=416):
        self._detector = DarknetObjectDetector(spec, weight, thresh, nms, draw)
        self.py_resize = py_resize
        self.net_size = net_size
    
    def format_image(self, pil_image):
        if self.py_resize:
            pil_image = pil_image.resize((self.net_size, self.net_size), Image.BILINEAR)
        
        data = np.array(pil_image).transpose([2,0,1]).astype(np.uint8).tostring()
        return data, (pil_image.size[0], pil_image.size[1])
    
    def detect_object(self, data, size):
        res = self._detector.detect_object(data, size[0], size[1], 3)
        out = [DetBBox(x) for x in res.content], res.load_time, res.pred_time
        if self.py_resize:
            print >> sys.stderr, "!! BBOX is in transformed dimensions -- need to implement fix"
            pass
        
        return out
    
    @staticmethod
    def set_device(gpu_id):
        DarknetObjectDetector.set_device(gpu_id)
