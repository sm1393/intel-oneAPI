#!/bin/bash

# python3 hybridnets_test_plot.py --use_optimization True --speed_test True

if [ -f images_with_optimization.txt ]; then
    rm images_with_optimization.txt
fi

output=$(python3 hybridnets_test_images.py --use_optimization True --speed_test True | grep "time")

if [ -n "$output" ]; then
    echo "$output" >> images_with_optimization.txt
fi

if [ -f images_without_optimization.txt ]; then
    rm images_without_optimization.txt
fi

output=$(python3 hybridnets_test_images.py --speed_test True | grep "time")

if [ -n "$output" ]; then
    echo "$output" >> images_without_optimization.txt
fi

python3 imageCount_vs_time_plot.py