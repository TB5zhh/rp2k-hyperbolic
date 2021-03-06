#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random

import shutil
import time
import warnings
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

from .cli import parse_args
from .data.rp2k import RP2kDataset
from .data.CIFAR100 import CIFAR100

best_acc1 = 0
train_step = 0
val_step = 0


def main():
    args = parse_args()
    print(f"Available GPU count: {torch.cuda.device_count()}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # freeze all layers but the last fc
    if args.require_grad == 'linear':
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
    # init the fc layer
    model.fc = torch.nn.Linear(in_features=2048, out_features=args.num_class, bias=True)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    linear_params = list(map(id, model.module.fc.parameters()))
    parameters = list(
        filter(lambda p: p.requires_grad and id(p) not in linear_params, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD([{
        'params': parameters,
        'lr': args.conv_lr
    }, {
        'params': model.module.fc.parameters(),
    }],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    if args.dataset == 'RP2k':
        train_dataset = RP2kDataset(
            '/root/rp2k/data',
            'train',
            args,
            args.shots,
            aug=[
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ],
        )  # For each class select up to 10 samples
        val_dataset = RP2kDataset(
            '/root/rp2k/data',
            'val',
            args,
            aug=[
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ],
        )
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(
            args.dataset_dir,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )
        val_dataset = CIFAR100(
            args.dataset_dir,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
        )

    train_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])
    val_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224),
    ])

    if args.dataset == 'RP2k':
        import json
        with open('/root/rp2k-moco/mapper.json', 'r') as f:
            mapper = torch.tensor(json.load(f))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, val_augmentation, args)
        return

    if args.wandb and args.rank == 0:
        wandb.init(project='hyp-moco-finetune', entity='air-sun')
        wandb.config.update(args)
        wandb.watch(model)
        wandb.run.name = args.run_name
        wandb.run.save()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, train_augmentation, args)

        if (epoch + 1) % 5 == 0:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, val_augmentation, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            # if not args.multiprocessing_distributed or (
            #         args.multiprocessing_distributed
            #         and args.rank % ngpus_per_node == 0):
            #     save_checkpoint(
            #         {
            #             'epoch': epoch + 1,
            #             'arch': args.arch,
            #             'state_dict': model.state_dict(),
            #             'best_acc1': best_acc1,
            #             'optimizer': optimizer.state_dict(),
            #         }, is_best)
            #     if epoch == args.start_epoch:
            #         sanity_check(model.state_dict(), args.pretrained)
    if args.distributed:
        dist.destroy_process_group()
    if args.wandb and args.rank == 0:
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, augment, args):
    global train_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    fine_top1 = AverageMeter('Fine-Acc@1', ':6.2f')
    fine_top5 = AverageMeter('Fine-Acc@5', ':6.2f')
    coarse_top1 = AverageMeter('Coarse-Acc@1', ':6.2f')
    coarse_top5 = AverageMeter('Coarse-Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, fine_top1, fine_top5, coarse_top1, coarse_top5],
        prefix="Epoch: [{}]".format(epoch))
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if args.require_grad == 'all':
        model.train()
        # model.eval()
    elif args.require_grad == 'linear':
        model.eval()
    else:
        raise NotImplementedError()

    end = time.time()
    for i, (image, fine_target, coarse_target) in enumerate(train_loader):
        train_step += 1
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
        fine_target = fine_target.cuda(args.gpu, non_blocking=True)
        coarse_target = coarse_target.cuda(args.gpu, non_blocking=True)

        image = augment(image)

        # compute output
        output = model(image)
        loss = criterion(output, fine_target)
        losses.update(loss.item(), image.size(0))

        # measure accuracy and record loss
        fine_acc1, fine_acc5 = accuracy(output, fine_target, topk=(1, 5))
        fine_top1.update(fine_acc1[0], image.size(0))
        fine_top5.update(fine_acc5[0], image.size(0))

        coarse_acc1, coarse_acc5 = accuracy(output,
                                            coarse_target,
                                            topk=(1, 5),
                                            apply=train_loader.dataset.map)
        coarse_top1.update(coarse_acc1[0], image.size(0))
        coarse_top5.update(coarse_acc5[0], image.size(0))

        if args.wandb and args.rank == 0:
            wandb.log({
                'loss': loss.item(),
                'loss_avg': losses.avg,
                'fine_acc1': fine_acc1[0].item(),
                'fine_acc5': fine_acc5[0].item(),
                'fine_acc1_avg': fine_top1.avg,
                'fine_acc5_avg': fine_top5.avg,
                'coarse_acc1': coarse_acc1[0].item(),
                'coarse_acc5': coarse_acc5[0].item(),
                'coarse_acc1_avg': coarse_top1.avg,
                'coarse_acc5_avg': coarse_top5.avg,
                'train_step': train_step,
            })

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, augment, args):
    global val_step
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    fine_top1 = AverageMeter('Acc@1', ':6.2f')
    fine_top5 = AverageMeter('Acc@5', ':6.2f')
    coarse_top1 = AverageMeter('Acc@1', ':6.2f')
    coarse_top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(len(val_loader),
                             [batch_time, losses, fine_top1, fine_top5, coarse_top1, coarse_top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (image, fine_target, coarse_target) in enumerate(val_loader):
            val_step += 1

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
            fine_target = fine_target.cuda(args.gpu, non_blocking=True)
            coarse_target = coarse_target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(image)
            loss = criterion(output, fine_target)
            losses.update(loss.item(), image.size(0))

            # measure accuracy and record loss
            # measure accuracy and record loss
            fine_acc1, fine_acc5 = accuracy(output, fine_target, topk=(1, 5))
            fine_top1.update(fine_acc1[0], image.size(0))
            fine_top5.update(fine_acc5[0], image.size(0))

            coarse_acc1, coarse_acc5 = accuracy(output,
                                                coarse_target,
                                                topk=(1, 5),
                                                apply=val_loader.dataset.map)
            coarse_top1.update(coarse_acc1[0], image.size(0))
            coarse_top5.update(coarse_acc5[0], image.size(0))
            if args.wandb and args.rank == 0:
                wandb.log({
                    'val_loss': loss.item(),
                    'val_loss_avg': losses.avg,
                    'val_fine_acc1': fine_acc1[0].item(),
                    'val_fine_acc5': fine_acc5[0].item(),
                    'val_fine_acc1_avg': fine_top1.avg,
                    'val_fine_acc5_avg': fine_top5.avg,
                    'val_coarse_acc1': coarse_acc1[0].item(),
                    'val_coarse_acc5': coarse_acc5[0].item(),
                    'val_coarse_acc1_avg': coarse_top1.avg,
                    'val_coarse_acc5_avg': coarse_top5.avg,
                    'val_step': val_step,
                })

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return fine_top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    conv_lr = args.conv_lr
    for milestone in args.schedule:
        lr *= 0.9 if epoch >= milestone else 1.
        conv_lr *= 0.9 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), apply=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        if apply is not None:
            pred = apply(pred)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
