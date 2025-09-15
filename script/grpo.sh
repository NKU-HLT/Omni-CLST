#!/bin/bash
name=$1
yaml_config=$2
timestamp=$3
number_gpu=$4
MASTER_PORT=$5
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
TRAIN_DEVICES=""
for i in "${!DEVICES[@]}"; do
    TRAIN_DEVICES+="${DEVICES[$i]},"
done

echo "Running $name with config $yaml_config"

NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES \
torchrun --nproc_per_node=$number_gpu \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --device_number $number_gpu \
    --yaml_config $yaml_config \
    --time $timestamp \