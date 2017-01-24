#!/bin/bash

cat ../pyDarknet/results/results-yolo-custom_* | fgrep flag | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/flag --class flag
cat ../pyDarknet/results/results-yolo-custom_* | fgrep explosion | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/explosion --class explosion
cat ../pyDarknet/results/results-yolo-custom_* | fgrep combat_vehicle | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/combat_vehicle --class combat_vehicle
cat ../pyDarknet/results/results-yolo-custom_* | fgrep long_gun | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/long_gun --class long_gun

tar -cf darknet-output.tar darknet-output

# SCP to local machine and annotate