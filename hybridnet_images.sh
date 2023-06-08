#!/bin/bash

# python3 hybridnets_test_plot.py --use_optimization True --speed_test True

if [ -f images_with_optimization_intel_np.txt ]; then
    rm images_with_optimization_intel_np.txt
fi

output=$(python3 hybridnets_test_plot.py --use_optimization True --speed_test True | grep "time")

if [ -n "$output" ]; then
    echo "$output" >> images_with_optimization_intel_np.txt
fi

if [ -f images_without_optimization_intel_np.txt ]; then
    rm images_without_optimization_intel_np.txt
fi

output=$(python3 hybridnets_test_plot.py --speed_test True | grep "time")

if [ -n "$output" ]; then
    echo "$output" >> images_without_optimization_intel_np.txt
fi

python3 imageCount_vs_time_plot.py