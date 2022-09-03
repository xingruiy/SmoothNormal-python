#!/bin/bash

scenes=(
    'office'
    'heads'
)

for scene in "${scenes[@]}"
do  
    # python data_gen/test/seven_scenes/gather_points.py \
    #     /mnt/dataset1/7-Scenes/$scene/seq-01 \
    #     data/seven_scenes/$scene/ --trainskip 10 --start 0 --end 1000
    python data_gen/test/generate_samples.py \
        data/seven_scenes/$scene/points.npz \
        --voxel_size 0.1 \
        --sample_std 0.015 \
        --down_ratio 0.01 
done