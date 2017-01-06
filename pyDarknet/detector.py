from libpydarknet import DarknetObjectDetector

from PIL import Image
import numpy as np
import time

class DetBBox(object):

    def __init__(self, bbox):
        self.left = bbox.left
        self.right = bbox.right
        self.top = bbox.top
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.cls = bbox.cls

class Darknet_ObjectDetector():

    def __init__(self, spec, weight, thresh=0.5, nms=0.4, draw=0):
        self._detector = DarknetObjectDetector(spec, weight, thresh, nms, draw)
    
    def format_image(self, pil_image):
        data = np.array(pil_image).transpose([2,0,1]).astype(np.uint8).tostring()
        return data, (pil_image.size[0], pil_image.size[1])
    
    def detect_object(self, data, size):
        start = time.time()
        rst = self._detector.detect_object(data, size[0], size[1], 3)
        end = time.time()
        ret_rst = [DetBBox(x) for x in rst]
        return ret_rst, end - start
    
    @staticmethod
    def set_device(gpu_id):
        DarknetObjectDetector.set_device(gpu_id)
