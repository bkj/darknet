#!/bin/bash

mkdir -p model
hadoop fs -copyToLocal /user/bjohnson/qcr/models/object_detection/darknet-0/* ./model
