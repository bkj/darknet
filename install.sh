#!/bin/bash

# --
# Install

# ... you may want to install this inside a conda env ...

sudo apt-get update

# Dependencies
sudo apt-get install -y libboost-all-dev
conda install -y opencv

rm -rf build
mkdir build
cd build
cmake ..
make -j12
