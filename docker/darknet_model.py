#!/usr/bin/env python

import os
import sys
import urllib
import contextlib
import cStringIO
import numpy as np

from glob import glob
from flask import abort
from PIL import Image

sys.path.append('/src/darknet/pyDarknet/')
from detector import Darknet_ObjectDetector as ObjectDetector
from detector import DetBBox

# --
# API

def url_to_image(url):
    try:
        with contextlib.closing(urllib.urlopen(url)) as req:
            local_url = cStringIO.StringIO(req).read())
        image = Image.open(local_url)
        if not image:
            abort(504)

        return image
    except:
        abort(504)

class apiModel():

    def __init__(self, model_path, model_name):
        self.model_name = model_name

        nms_thresh = 0.3
        conf_thresh = 0.5
        cfg_path = glob(os.path.join(model_path, '*cfg'))[0]
        weight_path = glob(os.path.join(model_path, '*weights'))[0]
        self.det = ObjectDetector(cfg_path, weight_path, conf_thresh, nms_thresh, 0)

        name_path = glob(os.path.join(model_path, '*names'))[0]
        self.class_names = open(name_path).read().splitlines()

    def _predict_api(self, url):
        image = url_to_image(url)
        if not np.any(image):
            return []

        image = self.det.format_image(image)
        detections, _, _ = self.det.detect_object(*image)

        results = []
        for bbox in detections:
            class_name = self.class_names[bbox.cls]
            results.append({
                "model" : self.model_name,
                "url" : url,
                "label" : class_name,
                "score" : round(float(bbox.confidence), 2),
                "bbox" : [
                      float(bbox.top),
                      float(bbox.left),
                      float(bbox.bottom),
                      float(bbox.right)
                ]
            })

        return results

    def predict_api(self, urls):
        urls = urls or []
        urls = filter(None, urls)
        if len(urls) == 0:
            return []

        results = [self._predict_api(url) for url in urls]
        return [obj for image in results for obj in image]
