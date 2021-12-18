WORKER=8
BATCHSIZE=256
LEARNINGRATE=0.001

python -m hyp2k.main_moco \
    --workers $WORKER \
    --batch-size $BATCHSIZE \
    --learning-rate $LEARNINGRATE \
    --rank 0 \
    --world-size 1 \
    --multiprocessing-distributed \
    --dist-url "tcp://127.0.0.1:12201" \
    --run-name "debug" \
    --hyper

    # --wandb \


