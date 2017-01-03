#!/usr/bin/env python

"""
	sloth2voc.py

	- Convert `sloth` JSON to `VOC` XML
    - Move images around
"""

import os
import cv2
import json
import shutil
import argparse
import numpy as np

from glob import glob
from collections import defaultdict

def make_xml(ann, im_path, source='sloth'):
    filename, boxes = ann['filename'], ann['annotations']
    height, width, depth = cv2.imread(im_path).shape
    header = (
        '<annotation>\n'
            '\t<folder>VOC2007</folder>\n'
            '\t<filename>{}</filename>\n'
            '\t<source>\n'
                '\t\t<database>The VOC2007 Database</database>\n'
                '\t\t<annotation>PASCAL VOC2007</annotation>\n'
                '\t\t<image>flickr</image>\n'
                '\t\t<flickrid>None</flickrid>\n'
            '\t</source>\n'
            '\t<owner>\n'
                '\t\t<flickrid>{}</flickrid>\n'
                '\t\t<name>?</name>\n'
            '\t</owner>\n'
            '\t<size>\n'
                '\t\t<width>{}</width>\n'
                '\t\t<height>{}</height>\n'
                '\t\t<depth>{}</depth>\n'
            '\t</size>\n'
            '\t<segmented>0</segmented>\n'
    ).format(
        os.path.basename(filename), 
        source, 
        width, 
        height, 
        depth
    )
    
    objs = ''
    for box in boxes:
        objs += (
            '\t<object>\n'
                '\t\t<name>{}</name>\n'
                '\t\t<pose>Unspecified</pose>\n'
                '\t\t<truncated>0</truncated>\n'
                '\t\t<difficult>0</difficult>\n'
                '\t\t<bndbox>\n'
                    '\t\t\t<xmin>{}</xmin>\n'
                    '\t\t\t<ymin>{}</ymin>\n'
                    '\t\t\t<xmax>{}</xmax>\n'
                    '\t\t\t<ymax>{}</ymax>\n'
                '\t\t</bndbox>\n'
            '\t</object>\n'
        ).format(
            box['class'],
            max(0, int(box['x'])),
            max(0, int(box['y'])),
            min(width, int(box['x']) + int(box['width'])),
            min(height, int(box['y']) + int(box['height']))
        )
    
    footer = '</annotation>'
    
    filename = os.path.basename(filename).split('.')
    filename[-1] = 'xml'
    filename = '.'.join(filename)
    return filename, header + objs + footer


def mkdirs(args):
    ann_out = os.path.join(args.outdir, 'annotations')
    img_out = os.path.join(args.outdir, 'images')
    set_out = os.path.join(args.outdir, 'image_sets')
    
    for path in [ann_out, img_out, set_out]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    return ann_out, img_out, set_out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    return parser.parse_args()

def merge_anns(all_anns):
    anns = defaultdict(list)
    for these_anns in all_anns:
        for a in these_anns:
            if len(a['annotations']) > 0:
                anns[a['filename']] += a['annotations']
    
    anns = [{'filename':k, 'class':'image', 'annotations':v} for k,v in anns.items()]
    return anns

if __name__ == "__main__":
    args = parse_args()
    ann_out, img_out, set_out = mkdirs(args)
    
    indirs = [x[0] for x in os.walk(args.indir)]
    indirs = filter(lambda x: os.path.exists(os.path.join(x, 'anns.json')), indirs)

    im_lookup = {}
    im_paths = np.hstack([glob(os.path.join(indir, '*')) for indir in indirs])
    for im_path in im_paths:
        im_lookup[os.path.basename(im_path)] = im_path

    all_anns = [json.load(open(os.path.join(indir, 'anns.json'))) for indir in indirs]
    anns = merge_anns(all_anns)
    
    for ann in anns:
        # Find source image
        im_path = im_lookup[ann['filename']]
        
        # Write annotations
        filename, xml = make_xml(ann, im_path)
        open(os.path.join(args.outdir, 'annotations', filename), 'w').write(xml)
        
        # Move images
        shutil.copy(im_path, os.path.join(img_out, ann['filename']))

