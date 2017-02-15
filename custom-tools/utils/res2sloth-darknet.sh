#!/bin/bash

INPATH='../pyDarknet/results/f-results-yolo-custom_final'

rm -r darknet-output

ANNPATH='/home/bjohnson/projects/py-faster-rcnn/custom-tools/output/eval-f/flag/anns.json'
cat $ANNPATH | jq --slurp -rc ".[] | .[] | if(.annotations | length > 0) then . else null end //empty | .filename" > .tmp
wc -l tmp
cat $INPATH | grep flag | sort -k3 -nr | grep -v -f .tmp |\
    ./utils/res2sloth-darknet.py --output ./darknet-output/f-final/flag --class flag

ANNPATH='/home/bjohnson/projects/py-faster-rcnn/custom-tools/output/eval-f/explosion/anns.json'
cat $ANNPATH | jq --slurp -rc ".[] | .[] | if(.annotations | length > 0) then . else null end //empty | .filename" > .tmp
wc -l tmp
cat $INPATH | grep explosion | sort -k3 -nr | grep -v -f .tmp |\
    ./utils/res2sloth-darknet.py --output ./darknet-output/f-final/explosion --class explosion

ANNPATH='/home/bjohnson/projects/py-faster-rcnn/custom-tools/output/eval-f/gun/anns.json'
cat $ANNPATH | jq --slurp -rc ".[] | .[] | if(.annotations | length > 0) then . else null end //empty | .filename" > .tmp
wc -l tmp
cat $INPATH | grep long_gun | sort -k3 -nr | grep -v -f .tmp |\
    ./utils/res2sloth-darknet.py --output ./darknet-output/f-final/long_gun --class long_gun

ANNPATH='/home/bjohnson/projects/py-faster-rcnn/custom-tools/output/eval-f/vehicle/anns.json'
cat $ANNPATH | jq --slurp -rc ".[] | .[] | if(.annotations | length > 0) then . else null end //empty | .filename" > .tmp
wc -l tmp
cat $INPATH | grep combat_vehicle | sort -k3 -nr | grep -v -f .tmp |\
    ./utils/res2sloth-darknet.py --output ./darknet-output/f-final/combat_vehicle --class combat_vehicle

rm .tmp

cd darknet-output
tar -cf darknet-f-final.tar f-final

# Annotate

