#!/bin/bash

MPATH="/home/bjohnson/projects/darknet-bkj/custom-tools/pfr-data/backup"

IMG_DIR=f
IMG_PATH="/home/bjohnson/projects/sm-analytics/etl/kafka/data/imgs/s3.amazonaws.com/pipeline-image-store/images/default/default/$IMG_DIR"

for weights in $(ls $MPATH | shuf); do
    f=$(echo $weights | cut -d'.' -f1)
    find $IMG_PATH -type f | ./detect-thread.py --weight-path $MPATH/$weights  > ./results/$IMT_DIR-results-$f
done

cat results/f-* | grep flag | awk -F'\t' '{OFS="\t"; print $1,$2,$3}' | sort -k3 -nr | head -n 1000 | cut -d$'\t' -f1 | sort | uniq | wc -l