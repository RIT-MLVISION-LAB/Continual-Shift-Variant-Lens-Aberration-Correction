#!/usr/bin/env bash

CONFIG=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

torchrun --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch