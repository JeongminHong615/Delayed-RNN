#!/bin/bash

export HYDRA_FULL_ERROR=1

GPU_ID=$1
MODEL=$2
DATASET=$3
HIDDEN_SIZE=$4
USE_WANDB=$5
GROUP_NAME=$6

echo "========================================"
echo "GPU: $GPU_ID | Model: $MODEL | Hidden: $HIDDEN_SIZE"
echo "========================================"

for SEED in 3 4 5; 
do
    echo "▶ Running SEED: $SEED ..."
    ARGS=(
        "seed=$SEED"  
        "model=$MODEL"
        "dataset=$DATASET"
        "hidden_size=$HIDDEN_SIZE"
        "wandb.use_wandb=$USE_WANDB"
        "wandb.group_name=$GROUP_NAME"
        "num_epochs=200"
    )
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python ../main.py "${ARGS[@]}"
done 
