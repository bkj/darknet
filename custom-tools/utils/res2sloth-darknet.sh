#!/bin/bash

cat ../results/comp4_det_test_flag.txt | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/flag --class flag
cat ../results/comp4_det_test_explosion.txt | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/explosion --class explosion
cat ../results/comp4_det_test_combat_vehicle.txt | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/combat_vehicle --class combat_vehicle
cat ../results/comp4_det_test_long_gun.txt | sort -k2 -nr | ./utils/res2sloth-darknet.py  --output ./darknet-output/long_gun --class long_gun
