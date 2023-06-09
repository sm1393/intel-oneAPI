#!/bin/bash

# python3 hybridnets_test_plot.py --use_optimization True --speed_test True

if [ -f video_with_optimization.txt ]; then
    rm video_with_optimization.txt
fi

output=$(python3 hybridnets_test_videos.py --use_optimization True | grep "time")

if [ -n "$output" ]; then
    echo "$output" >> video_with_optimization.txt
fi

if [ -f video_without_optimization.txt ]; then
    rm video_without_optimization.txt
fi

output=$(python3 hybridnets_test_videos.py | grep "time")

if [ -n "$output" ]; then
    echo "$output" >> video_without_optimization.txt
fi

python3 frameCount_vs_time_plot.py