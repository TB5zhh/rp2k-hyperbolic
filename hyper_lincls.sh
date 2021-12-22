#!/bin/bash
set -e

BATCH_SIZE=256
CONV_LR=1e-3
LINEAR_LR=3
RUN_NAME=$1

if [ ! -n "$RUN_NAME" ]; then
    echo "You must manually specify the run name to start the experiments"
    exit -1
fi

python -m hyp2k.main_lincls \
    -a resnet50 \
    --conv_lr 1e-3 \
    --lr 3 \
    --batch-size 256 \
    --dist-url 'tcp://localhost:11111' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --run-name "$RUN_NAME" \
    --require_grad all \
    --dataset "cifar100" \
    --dataset-dir "/home/aidrive/tb5zhh/data/" \
    --pretrained "checkpoint_MoCo on CIFAR100 test_0195.pth.tar" \
    --shots 20 \
    --num-class 100 \
    --wandb \
    --moco-dim 256


