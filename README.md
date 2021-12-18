## Installation

```shell
conda create -n rp2k-hyper python=3.8
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install wandb ipykernel yapf

cd hyperbolic-image-embeddings
python setup.py install

```