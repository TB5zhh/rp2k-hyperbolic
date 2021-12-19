## Installation

```shell
conda create -n rp2k-hyper python=3.8
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install wandb ipykernel yapf

cd hyperbolic-image-embeddings
python setup.py install

```

## Disc

`checkpoints/checkpoint_debug_00xx.pth.tar` is the hyperbolic pretrain checkpoint at #xx epoch. 

The wandb dashboard of hyperbolic pretraining can be found at https://wandb.ai/air-sun/hyp-moco/runs/2r7ya2e5