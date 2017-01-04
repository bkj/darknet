#!/bin/bash

cat ../results/comp4_det_test_flag.txt | ./utils/res2sloth-darknet.py  --output ./output/flag --class flag
cat ../results/comp4_det_test_explosion.txt | ./utils/res2sloth-darknet.py  --output ./output/explosion --class explosion
cat ../results/comp4_det_test_combat_vehicle.txt | ./utils/res2sloth-darknet.py  --output ./output/combat_vehicle --class combat_vehicle
cat ../results/comp4_det_test_long_gun.txt | ./utils/res2sloth-darknet.py  --output ./output/long_gun --class long_gun

