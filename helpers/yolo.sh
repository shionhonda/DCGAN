#!/bin/sh

for file in ../images/dressed/*.jpg
do
    ../../darknet/darknet detect ../../darknet/cfg/yolov3.cfg ../../darknet/yolov3.weights $file
done
