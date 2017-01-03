#!/usr/bin/env python

"""
    darknet-label.py
"""

import argparse
from glob import glob

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(ann_path, label_path, image_id):
    in_file = open(os.path.join(ann_path, image_id + '.xml'))
    out_file = open(os.path.join(label_path, image_id + '.txt'), 'w')
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('ymax').text)
        )
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def _get_classes(ann_path):
    print '... inferring classes ... '
    classes = set([])
    for ann in glob(ann_path + '/*xml'):
        xml = ET.parse(ann)
        for x in xml.findall('object'):
            classes.add(x.find('name').text)
    
    return sorted(list(classes))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='./pfr-data/')
    return parser.parse_args()

# --

if __name__ == "__main__":
    wd = getcwd()
    args = parse_args()
    
    ann_path = os.path.join(args.indir, 'annotations')
    img_path = os.path.join(args.indir, 'images')
    set_path = os.path.join(args.indir, 'image_sets')
    
    name_path = os.path.join(args.indir, 'custom.names')
    data_path = os.path.join(args.indir, 'custom.data')
    lab_path = os.path.join(args.indir, 'labels')
    list_path = os.path.join(args.indir, 'darknet-image_sets')
    backup_path = os.path.join(args.indir, 'backup')
    
    # Write names to file
    classes = _get_classes(ann_path)
    print classes
    open(name_path, 'w').write('\n'.join(classes) + '\n')
    
    # Create directories
    for path in [lab_path, list_path, backup_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Write train/test files and create `darknet` annotation format
    for mode in ['trainval', 'test']:
        list_file = open(os.path.join(list_path, mode + '.txt'), 'w')
        image_filenames = open(os.path.join(set_path, mode + '.txt')).read().strip().split()
        for image_filename in image_filenames:
            list_file.write(os.path.join(wd, img_path, image_filename) + '\n')
            convert_annotation(ann_path, lab_path, image_filename.split('.')[0])
        
    list_file.close()
    
    # Write `.data` file
    with open(data_path, 'w') as data_file:
        data_file.write('classes = %d\n' % len(classes))
        data_file.write('train = %s\n' % os.path.abspath(os.path.join(list_path, 'trainval' + '.txt')))
        data_file.write('valid = %s\n' % os.path.abspath(os.path.join(list_path, 'test' + '.txt')))
        data_file.write('names = %s\n' % os.path.abspath(name_path))
        data_file.write('backup = %s\n' % os.path.abspath(backup_path))
    
    data_file.close()
    