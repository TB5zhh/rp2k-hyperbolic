import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet50',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('-j',
                        '--workers',
                        default=32,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.03,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--schedule',
                        default=[120, 160],
                        nargs='*',
                        type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco-dim',
                        default=128,
                        type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument(
        '--moco-k',
        default=65536,
        type=int,
        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument(
        '--moco-m',
        default=0.999,
        type=float,
        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t',
                        default=0.07,
                        type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true', help='use mlp head')
    parser.add_argument('--aug-plus',
                        action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos',
                        action='store_true',
                        help='use cosine lr schedule')

    # options for rp2k
    parser.add_argument('--load-all',
                        action='store_true',
                        help='load all images into memory at once')
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--expo',
                        action='store_true',
                        help='exponential scheduler ')
    parser.add_argument('--hyper',
                        action='store_true',
                        help='use hyperbolic momentum contrast')
    parser.add_argument('--run-name',
                        type=str,
                        default='train',
                        help='exponential scheduler ')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar100',
                        help='The dataset used in pipeline')
    parser.add_argument('--dataset-dir',
                        type=str,
                        default='/',
                        help='The dataset path')
    parser.add_argument('--pretrained',
                        default='',
                        type=str,
                        help='path to moco pretrained checkpoint')

    parser.add_argument('--require_grad',
                        type=str,
                        default='linear',
                        help='Train all network or train the linear layer only')
    parser.add_argument('--conv_lr', type=float, default=1e-3)
    parser.add_argument('--shots', type=int, default=10)
    parser.add_argument('--num-class', type=int, default=2388)
    
    parser.add_argument('--riemannian',
                        action='store_true',
                        help='use riemannian grad fix in hyperbolic space')
    parser.add_argument('--train-c',
                        action='store_true',
                        help='train c in hyperbolic space')
    parser.add_argument('--train-x',
                        action='store_true',
                        help='train x in hyperbolic space')
    parser.add_argument('--c',
                        type=float,
                        default=1.0,
                        help='negative curvature')
    return parser.parse_args()