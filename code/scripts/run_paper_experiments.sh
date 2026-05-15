#!/bin/bash
# Reproduce all experiments reported in the paper.
#
# Usage:
#   bash scripts/run_paper_experiments.sh [GPU_ID]
#
# Runs the fixed-length N=50 DelaySequence task with dynamic delays
# d ~ Uniform{5,...,20} for LSTM / Transformer / LD-RNN.

set -e
export HYDRA_FULL_ERROR=1
GPU_ID=${1:-0}

run() {
    local model="$1"; shift
    echo "==== $model ===="
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        model=$model dataset=delaysequence seed=0 \
        num_epochs=50 hidden_size=64 lr=0.01 \
        dataset.min_len=50 dataset.max_len=50 \
        +dataset.min_delay=5 +dataset.max_delay=20 \
        "$@"
}

run RNN
run GRU
run LSTM
run Transformer
run DRNN_jm model.max_delay=75
