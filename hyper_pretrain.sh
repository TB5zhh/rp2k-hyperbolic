#!/bin/bash
set -e

WORKER=8
BATCHSIZE=128
LR=0.01
RUN_NAME=$1

if [ ! -n "$RUN_NAME" ]; then
    echo "You must manually specify the run name to start the experiments"
    exit -1
fi


python -m hyp2k.main_moco \
    --workers $WORKER \
    --batch-size $BATCHSIZE \
    --learning-rate $LR \
    --rank 0 \
    --world-size 1 \
    --multiprocessing-distributed \
    --dist-url "tcp://127.0.0.1:12201" \
    --run-name "$RUN_NAME" \
    --dataset "cifar100" \
    --dataset-dir "/home/aidrive/tb5zhh/data/" \
    --hyper \
    --moco-dim 256


    # --wandb \
