#!/bin/bash

# train.sh
#
# Train `yolo` model

# !! Should deploy data for negative annotations (eg empty images)
# !! Should remove all mortar annotations (doesn't really work)

# Convert `sloth` format to `VOC` and copy annotated images to directory
./utils/sloth2voc.py --indir ./source-data --outdir ./pfr-data

# Make train/test split
./utils/train-test-split.py --indir ./pfr-data --keep-extensions

# Make train/test files and annotation files
./utils/deploy-dataset-darknet.py --indir ./pfr-data

# Make model file
cp ../cfg/yolo-voc.cfg ./pfr-data/yolo-custom.cfg
N=$(cat ./pfr-data/custom.names | wc -l)
cat ./pfr-data/yolo-custom.cfg | sed "s/classes=20/classes=$N/" > tmp && mv tmp ./pfr-data/yolo-custom.cfg
# !! Also change number of filters in last conv layer to `(num_classes + coords + 1) * num`
# Eg (5 + 4 + 1) * 5 = 50 for 5 classes

cd ../build
./darknet detector train \
    ../custom-tools/pfr-data/custom.data \
    ../custom-tools/pfr-data/yolo-custom.cfg \
    ../darknet19_448.conv.23