WORKER=8
BATCHSIZE=128
LEARNINGRATE=0.01

python -m hyp2k.main_moco \
    --workers $WORKER \
    --batch-size $BATCHSIZE \
    --learning-rate $LEARNINGRATE \
    --rank 0 \
    --world-size 1 \
    --multiprocessing-distributed \
    --dist-url "tcp://127.0.0.1:12201" \
    --run-name "debug" \
    --wandb \
    --dataset "cifar100" \
    --dataset-dir "/home/aidrive/tb5zhh/data/cifar-100-python"



