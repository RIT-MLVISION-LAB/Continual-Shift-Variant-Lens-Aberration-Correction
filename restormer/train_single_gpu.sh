#!/usr/bin/env bash
# Single-GPU training script for Restormer
# Usage: ./train_single_gpu.sh <config.yml>

CONFIG=$1

if [ -z "$CONFIG" ]; then
    echo "Usage: ./train_single_gpu.sh <config.yml>"
    echo "Example: ./train_single_gpu.sh Motion_Deblurring/Options/Shift_Variant_V1_Deblurring_Restormer.yml"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Starting single-GPU training with config: $CONFIG"
python basicsr/train.py -opt $CONFIG
