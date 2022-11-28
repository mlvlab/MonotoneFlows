import argparse
import logging
import time
import math
import datetime
import os
import os.path
import sys
import numpy as np
from tqdm import tqdm
import gc
import copy

os.environ['NVIDIA_TF32_OVERRIDE']='0'

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets
import torch.multiprocessing as mp

from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts


'''
Utility functions
'''
def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def add_noise(x, apply=True, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if apply:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x


'''
Training routine
'''
def main_worker(rank, world_size, port, args_dict):
    global args
    args = args_dict

    # disable TF32 in favor of FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # logger
    if rank == 0:
        utils.makedirs(args.save)
        logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    else:
        logger = logging.getLogger()
    logger.info(args)
    
    # init process group
    if args.distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '%d' % (port)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    if args.distributed:
        device = torch.device('cuda')
        torch.cuda.set_device(rank)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
    else:
        logger.info('WARNING: Using device {}'.format(device))

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)

    if args.var_deq:
        print('Setting --add-noise option to False (currently set as %s) as variational dequantization is enabled.' % (str(args.add_noise)))
        args.add_noise = False

    logger.info('Loading dataset {}'.format(args.data))
    # Dataset and hyperparameters
    if args.data == 'cifar10':
        im_dim = 3
        n_classes = 10
        if args.task in ['classification', 'hybrid']:

            # Classification-specific preprocessing.
            transform_train = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: add_noise(x, apply=args.add_noise),
            ])

            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                lambda x: add_noise(x, apply=args.add_noise),
            ])

            # Remove the logit transform.
            init_layer = layers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            transform_train = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: add_noise(x, apply=args.add_noise),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                lambda x: add_noise(x, apply=args.add_noise),
            ])
            init_layer = layers.LogitTransform(0.05)
        if args.distributed and rank != 0:
            torch.distributed.barrier()
        train_dataset = datasets.CIFAR10(args.dataroot, train=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(args.dataroot, train=False, transform=transform_test)
        if args.distributed and rank == 0:
            torch.distributed.barrier()
    elif args.data == 'mnist':
        im_dim = 1
        init_layer = layers.LogitTransform(1e-6)
        n_classes = 10
        if args.distributed and rank != 0:
            torch.distributed.barrier()
        train_dataset = datasets.MNIST(
                args.dataroot, train=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    lambda x: add_noise(x, apply=args.add_noise),
                ])
            )
        test_dataset = datasets.MNIST(
                args.dataroot, train=False, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    lambda x: add_noise(x, apply=args.add_noise),
                ])
            )
        if args.distributed and rank == 0:
            torch.distributed.barrier()
    elif args.data == 'svhn':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        n_classes = 10
        if args.distributed and rank != 0:
            torch.distributed.barrier()
        train_dataset = vdsets.SVHN(
                args.dataroot, split='train', download=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.RandomCrop(32, padding=4, padding_mode=args.rcrop_pad_mode),
                    transforms.ToTensor(),
                    lambda x: add_noise(x, apply=args.add_noise),
                ])
            )
        test_dataset = vdsets.SVHN(
                args.dataroot, split='test', download=True, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    lambda x: add_noise(x, apply=args.add_noise),
                ])
            )
        if args.distributed and rank == 0:
            torch.distributed.barrier()
    elif args.data == 'celebahq':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 256:
            logger.info('Changing image size to 256.')
            args.imagesize = 256
        train_dataset = datasets.CelebAHQ(
                train=True, transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    reduce_bits,
                    lambda x: add_noise(x, apply=args.add_noise, nvals=2**args.nbits),
                ])
            )
        test_dataset = datasets.CelebAHQ(
                train=False, transform=transforms.Compose([
                    reduce_bits,
                    lambda x: add_noise(x, apply=args.add_noise, nvals=2**args.nbits),
                ])
            )
    elif args.data == 'celeba_5bit':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 64:
            logger.info('Changing image size to 64.')
            args.imagesize = 64
        train_dataset = datasets.CelebA5bit(
                train=True, transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    lambda x: add_noise(x, apply=args.add_noise, nvals=32),
                ])
            )
        test_dataset = datasets.CelebA5bit(train=False, transform=transforms.Compose([
                lambda x: add_noise(x, apply=args.add_noise, nvals=32),
            ]))
    elif args.data == 'imagenet32':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 32:
            logger.info('Changing image size to 32.')
            args.imagesize = 32
        train_dataset = datasets.Imagenet32(train=True, transform=transforms.Compose([
                lambda x: add_noise(x, apply=args.add_noise),
            ]))
        test_dataset = datasets.Imagenet32(train=False, transform=transforms.Compose([
                lambda x: add_noise(x, apply=args.add_noise),
            ]))
    elif args.data == 'imagenet64':
        im_dim = 3
        init_layer = layers.LogitTransform(0.05)
        if args.imagesize != 64:
            logger.info('Changing image size to 64.')
            args.imagesize = 64
        train_dataset = datasets.Imagenet64(train=True, transform=transforms.Compose([
                lambda x: add_noise(x, apply=args.add_noise),
            ]))
        test_dataset = datasets.Imagenet64(train=False, transform=transforms.Compose([
                lambda x: add_noise(x, apply=args.add_noise),
            ]))


    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )

    mp_context = torch.multiprocessing.get_context('fork') if args.nworkers > 0 else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        num_workers=args.nworkers,
        sampler=train_sampler,
        multiprocessing_context=mp_context,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
        multiprocessing_context=mp_context,
    )


    if args.task in ['classification', 'hybrid']:
        try:
            n_classes
        except NameError:
            raise ValueError('Cannot perform classification with {}'.format(args.data))
    else:
        n_classes = 1

    logger.info('Dataset loaded.')
    logger.info('Creating model.')

    input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)
    dataset_size = len(train_loader.dataset)

    if args.squeeze_first:
        input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
        squeeze_layer = layers.SqueezeLayer(2)

    if args.act in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU'] or (args.var_deq and args.var_deq_act in ['CLipSwish', 'CPila', 'ALCLipSiLU', 'CReLU']):
        assert (args.densenet_growth % 2 == 0 | args.fc_densenet_growth % 2 == 0), "Select an even densenet growth size!"

    # Model
    model = ResidualFlow(
        input_size,
        n_blocks=list(map(int, args.nblocks.split('-'))),
        intermediate_dim=args.idim,
        factor_out=args.factor_out,
        quadratic=args.quadratic,
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_actnorm=args.fc_actnorm,
        batchnorm=args.batchnorm,
        dropout=args.dropout,
        fc=args.fc,
        coeff=args.coeff,
        vnorms=args.vnorms,
        n_lipschitz_iters=args.n_lipschitz_iters,
        sn_atol=args.sn_tol,
        sn_rtol=args.sn_tol,
        n_power_series=args.n_power_series,
        n_dist=args.n_dist,
        n_samples=args.n_samples,
        kernels=args.kernels,
        activation_fn=args.act,
        fc_end=args.fc_end,
        fc_idim=args.fc_idim,
        n_exact_terms=args.n_exact_terms,
        preact=args.preact,
        neumann_grad=args.neumann_grad,
        grad_in_forward=args.mem_eff,
        first_resblock=args.first_resblock,
        learn_p=args.learn_p,
        classification=args.task in ['classification', 'hybrid'],
        classification_hdim=args.cdim,
        n_classes=n_classes,
        block_type=args.block,
        densenet=args.densenet,
        densenet_depth=args.densenet_depth,
        densenet_growth=args.densenet_growth,
        fc_densenet_growth=args.fc_densenet_growth,
        learnable_concat=args.learnable_concat,
        lip_coeff=args.lip_coeff,
        monotone_resolvent=args.monotone_resolvent,
        var_deq=args.var_deq,
        var_deq_nblocks=args.var_deq_nblocks,
        var_deq_act=args.var_deq_act,
        var_deq_mf=args.var_deq_mf,
        var_deq_nbits=args.nbits
    )

    model.to(device)
    ema = utils.ExponentialMovingAverage(model)

    def parallelize(model):
        if args.distributed:
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)
        else:
            return model

    logger.info(model)
    logger.info('EMA: {}'.format(ema))

    best_test_bpd = math.inf
    if (args.resume is not None):
        logger.info('Resuming model from {}'.format(args.resume))
        with torch.no_grad():
            x = torch.rand(1, *input_size[1:]).to(device)
            model(x)

        ### CPU or GPU choice
        if torch.cuda.is_available() is False:
            checkpt = torch.load(args.resume, map_location=torch.device('cpu'))
        else:
            checkpt = torch.load(args.resume, map_location=torch.device(torch.cuda.current_device()))

        begin_epoch = checkpt['begin_epoch']
        begin_iter = checkpt['begin_iter']
    else:
        begin_epoch = 0
        begin_iter = 0

    scheduler = None

    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd, foreach=True)
        if args.scheduler: scheduler = CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2, last_epoch=begin_epoch - 1)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd, foreach=True)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model_params, lr=args.lr, weight_decay=args.wd, foreach=True)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.wd, foreach=True)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=begin_epoch - 1
            )
    else:
        raise ValueError('Unknown optimizer {}'.format(args.optimizer))

    if (args.resume is not None):
        sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
        state = model.state_dict()
        state.update(sd)
        model.load_state_dict(state, strict=True)
        ema.set(checkpt['ema'])
        if 'optimizer_state_dict' in checkpt:
            optimizer.load_state_dict(checkpt['optimizer_state_dict'])
            # Manually move optimizer state to GPU
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        del checkpt
        del state

    logger.info(optimizer)

    if rank == 0:
        fixed_z = standard_normal_sample([min(32, args.batchsize),
                                          (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)

    criterion = torch.nn.CrossEntropyLoss()


    def compute_loss(x, model, beta=1.0):
        bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(n_classes).to(x)
        logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

        if args.data == 'celeba_5bit':
            nvals = 32
        elif args.data == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256

        x, logpu = add_padding(x, nvals)

        if args.squeeze_first:
            x = squeeze_layer(x)

        if args.task == 'hybrid':
            z_logp, logits_tensor = model(x.view(-1, *input_size[1:]), 0, classify=True)
            z, delta_logp = z_logp
        elif args.task == 'density':
            z, delta_logp = model(x.view(-1, *input_size[1:]), 0)
        elif args.task == 'classification':
            z, logits_tensor = model(x.view(-1, *input_size[1:]), classify=True)

        if args.task in ['density', 'hybrid']:
            # log p(z)
            logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

            # log p(x)
            logpx = logpz - beta * delta_logp - np.log(nvals) * (
                args.imagesize * args.imagesize * (im_dim + args.padding)
            ) - logpu
            bits_per_dim = -torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim) / np.log(2)

            logpz = torch.mean(logpz).detach()
            delta_logp = torch.mean(-delta_logp).detach()

        return bits_per_dim, logits_tensor, logpz, delta_logp


    def estimator_moments(model, baseline=0):
        avg_first_moment = 0.
        avg_second_moment = 0.
        for m in model.modules():
            if isinstance(m, layers.iResBlock) or isinstance(m, layers.VarDeqBlock):
                avg_first_moment += m.last_firmom.item()
                avg_second_moment += m.last_secmom.item()
        return avg_first_moment, avg_second_moment


    def compute_p_grads(model):
        scales = 0.
        nlayers = 0
        for m in model.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                scales = scales + m.compute_one_iter()
                nlayers += 1
        scales.mul(1 / nlayers).backward()
        for m in model.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                if m.domain.grad is not None and torch.isnan(m.domain.grad):
                    m.domain.grad = None


    batch_time = utils.RunningAverageMeter(0.97)
    bpd_meter = utils.RunningAverageMeter(0.97)
    logpz_meter = utils.RunningAverageMeter(0.97)
    deltalogp_meter = utils.RunningAverageMeter(0.97)
    firmom_meter = utils.RunningAverageMeter(0.97)
    secmom_meter = utils.RunningAverageMeter(0.97)
    gnorm_meter = utils.RunningAverageMeter(0.97)
    ce_meter = utils.RunningAverageMeter(0.97)


    def train(epoch, _begin_iter, _model):
        total = 0
        correct = 0

        end = time.time()

        if epoch == 0 and _begin_iter == 0:
            _model.train()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                if args.squeeze_first:
                    x = squeeze_layer(x)
                z = _model.forward(x)
                del z
                break

        if _begin_iter == 0:
            update_lipschitz(_model)

        if epoch == 0 and _begin_iter == 0:
            for m in _model.modules():
                if isinstance(m, layers.ActNormNd):
                    m.initialized.data.zero_()

        _model_p = parallelize(_model)
        _model_p.train()

        train_sampler.set_epoch(epoch)
        for i, (x, y) in enumerate(train_loader):
            if i < _begin_iter:
                continue

            global_itr = epoch * len(train_loader) + i
            update_lr(optimizer, global_itr)

            # Training procedure:
            # for each sample x:
            #   compute z = f(x)
            #   maximize log p(x) = log p(z) - log |det df/dx|

            x = x.to(device)

            beta = beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
            bpd, logits, logpz, neg_delta_logp = compute_loss(x, _model_p, beta=beta)

            with torch.no_grad():
                if args.task in ['density', 'hybrid']:
                    firmom, secmom = estimator_moments(_model)
                    
                    ts = torch.stack([bpd, logpz, neg_delta_logp, torch.tensor(firmom, device=bpd.device), torch.tensor(secmom, device=bpd.device)], dim=0)
                    if args.distributed:
                        torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.SUM)
                        ts = ts / world_size
    
                    bpd_meter.update(ts[0].item())
                    logpz_meter.update(ts[1].item())
                    deltalogp_meter.update(ts[2].item())
                    firmom_meter.update(ts[3].item())
                    secmom_meter.update(ts[4].item())
    
            if args.task in ['classification', 'hybrid']:
                y = y.to(device)
                crossent = criterion(logits, y)
                
                ts = torch.stack([crossent], dim=0)
                if args.distributed:
                    torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.SUM)
                    ts = ts / world_size
                ce_meter.update(ts[0].item())

                # Compute accuracy.
                _, predicted = logits.max(1)
                ts = torch.stack([torch.tensor(y.size(0), device=bpd.device), predicted.eq(y).sum()], dim=0)
                if args.distributed:
                    torch.distributed.all_reduce(ts, op=torch.distributed.ReduceOp.SUM)
                    ts = ts / world_size
                total += ts[0].item()
                correct += ts[1].item()

            # compute gradient and do SGD step
            if args.task == 'density':
                loss = bpd
            elif args.task == 'classification':
                loss = crossent
            else:
                if not args.scale_dim: bpd = bpd * (args.imagesize * args.imagesize * im_dim)
                loss = bpd + crossent / np.log(2)  # Change cross entropy from nats to bits.
            loss.backward()

            if global_itr % args.update_freq == args.update_freq - 1:

                if args.update_freq > 1:
                    with torch.no_grad():
                        for p in _model.parameters():
                            if p.grad is not None:
                                p.grad /= args.update_freq

                if args.clip_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(_model.parameters(), 1.)
                else:
                    grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(_model.parameters(), 1e8)

                if args.learn_p: compute_p_grads(_model)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Start learning concat after X epochs
                if args.learnable_concat and (epoch < args.start_learnable_concat):
                    reset_parameters(_model)

                update_lipschitz(_model)
                ema.apply()

                gnorm_meter.update(grad_norm)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                s = (
                    'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                    'GradNorm {gnorm_meter.avg:.2f}'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
                    )
                )

                if args.task in ['density', 'hybrid']:
                    s += (
                        ' | Bits/dim {bpd_meter.val:.4f}({bpd_meter.avg:.4f}) | '
                        'Logpz {logpz_meter.avg:.0f} | '
                        '-DeltaLogp {deltalogp_meter.avg:.0f} | '
                        'EstMoment ({firmom_meter.avg:.0f},{secmom_meter.avg:.0f})'.format(
                            bpd_meter=bpd_meter, logpz_meter=logpz_meter, deltalogp_meter=deltalogp_meter,
                            firmom_meter=firmom_meter, secmom_meter=secmom_meter
                        )
                    )

                if args.task in ['classification', 'hybrid']:
                    s += ' | CE {ce_meter.avg:.4f} | Acc {0:.4f}'.format(100 * correct / total, ce_meter=ce_meter)

                logger.info(s)

            if rank == 0 and i % args.vis_freq == 0:
                visualize(epoch, _model, i, x)

            if args.save_freq > 0 and i % args.save_freq == args.save_freq - 1:
                save_dir = os.path.join(args.save, 'models')
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'state_dict': _model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'ema': ema,
                    'test_bpd': 0.0,
                    'begin_epoch': epoch,
                    'begin_iter': i + 1,
                }, os.path.join(save_dir, 'most_recent_intraepoch.pth'))

            del x



    def validate(epoch, _model_orig, ema=None):
        """
        Evaluates the cross entropy between p_data and p_model.
        """
        bpd_meter = utils.AverageMeter()
        ce_meter = utils.AverageMeter()

        if ema is not None:
            ema.swap()

        _model = copy.deepcopy(_model_orig)
    
        update_lipschitz(_model)

        '''
        During validation, we do not use DDP for parallelism. We instead use a bare bone workload distribution.
        '''
        #model = parallelize(model)
        _model.eval()

        correct = 0
        total = 0

        if args.distributed:
            torch.distributed.barrier()

        start = time.time()
        with torch.no_grad():
            test_loader_tqdm = tqdm(test_loader) if rank == 0 else test_loader
            for i, (x, y) in enumerate(test_loader_tqdm):
                if (i % world_size == rank):
                    x = x.to(device)
                    bpd, logits, _, _ = compute_loss(x, _model)
                    bpd_meter.update(bpd.item(), x.size(0))

                    if args.task in ['classification', 'hybrid']:
                        y = y.to(device)
                        loss = criterion(logits, y)
                        ce_meter.update(loss.item(), x.size(0))
                        _, predicted = logits.max(1)
                        total += y.size(0)
                        correct += predicted.eq(y).sum().item()
        if args.distributed:
            torch.distributed.barrier()
        val_time = time.time() - start

        if ema is not None:
            ema.swap()

        bpd_meter_tensor = torch.tensor([bpd_meter.sum, bpd_meter.count], dtype=torch.float, device=device)
        if args.distributed:
            torch.distributed.all_reduce(bpd_meter_tensor, op=torch.distributed.ReduceOp.SUM)
        bpd_meter_avg = (bpd_meter_tensor[0]/bpd_meter_tensor[1]).item()

        s = 'Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {2:.4f}'.format(epoch, val_time, bpd_meter_avg)
        if args.task in ['classification', 'hybrid']:
            ce_meter_tensor = torch.tensor([ce_meter.sum, ce_meter.count, correct, total], dtype=torch.float, device=device)
            if args.distributed:
                torch.distributed.all_reduce(ce_meter_tensor, op=torch.distributed.ReduceOp.SUM)
            ce_meter_avg = (ce_meter_tensor[0]/ce_meter_tensor[1]).item()
            acc_avg = 100 * (ce_meter_tensor[2]/ce_meter_tensor[3]).item()
            s += ' | CE {:.4f} | Acc {:.2f}'.format(ce_meter_avg, acc_avg)
        logger.info(s)
        return bpd_meter_avg


    def visualize(epoch, _model, itr, real_imgs):
        _model.eval()
        utils.makedirs(os.path.join(args.save, 'imgs'))
        real_imgs = real_imgs[:32]
        _real_imgs = real_imgs

        if args.data == 'celeba_5bit':
            nvals = 32
        elif args.data == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256

        with torch.no_grad():
            # reconstructed real images
            real_imgs, _ = add_padding(real_imgs, nvals)
            if args.squeeze_first: real_imgs = squeeze_layer(real_imgs)
            recon_imgs = _model(_model(real_imgs.view(-1, *input_size[1:])), inverse=True).view(-1, *input_size[1:])
            if args.squeeze_first: recon_imgs = squeeze_layer.inverse(recon_imgs)
            recon_imgs = remove_padding(recon_imgs)

            # random samples
            fake_imgs = _model(fixed_z, inverse=True).view(-1, *input_size[1:])
            if args.squeeze_first: fake_imgs = squeeze_layer.inverse(fake_imgs)
            fake_imgs = remove_padding(fake_imgs)

            fake_imgs = fake_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
            recon_imgs = recon_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
            imgs = torch.cat([_real_imgs, fake_imgs, recon_imgs], 0)

            filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
            save_image(imgs.cpu().float(), filename, nrow=16, padding=2)
        _model.train()


    def get_lipschitz_constants(model):
        lipschitz_constants = []
        for m in model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                lipschitz_constants.append(m.scale)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                lipschitz_constants.append(m.scale)
            if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
                lipschitz_constants.append(m.scale)
        return lipschitz_constants


    def get_learnable_params(model):
        concat_eta1 = []
        concat_eta2 = []
        concat_K1 = []
        concat_K2 = []
        for m in model.modules():
            if isinstance(m, layers.LipschitzDenseLayer):
                eta1_normalized, eta2_normalized = m.get_eta1_eta2()
                concat_eta1.append(eta1_normalized.item())
                concat_eta2.append(eta2_normalized.item())

                K1_unnormalized = m.K1_unnormalized
                K2_unnormalized = m.K2_unnormalized
                concat_K1.append(K1_unnormalized.item())
                concat_K2.append(K2_unnormalized.item())
        return concat_eta1, concat_eta2, concat_K1, concat_K2


    def get_activation_params(model):
        alphas = []
        betas = []
        for m in model.modules():
            if isinstance(m, layers.base.activations.LeakyLSwish):
                alpha = m.alpha
                beta = m.beta
                alphas.append(round(alpha.item(), 2))
                betas.append(round(beta.item(), 2))
        return alphas, betas


    def reset_parameters(model):
        for m in model.modules():
            if isinstance(m, layers.LipschitzDenseLayer):
                torch.nn.init.ones_(m.K1_unnormalized)
                torch.nn.init.ones_(m.K2_unnormalized)


    def update_lipschitz(model):
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                    m.compute_weight(update=True)
                if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                    m.compute_weight(update=True)


    def get_ords(model):
        ords = []
        for m in model.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                domain, codomain = m.compute_domain_codomain()
                if torch.is_tensor(domain):
                    domain = domain.item()
                if torch.is_tensor(codomain):
                    codomain = codomain.item()
                ords.append(domain)
                ords.append(codomain)
        return ords


    def pretty_repr(a):
        return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'

    #global best_test_bpd

    last_checkpoints = []
    lipschitz_constants = []
    ords = []
    alphas = []
    betas = []
    concat_eta1 = []
    concat_eta2 = []
    concat_K1 = []
    concat_K2 = []

    # if args.resume:
    #    validate(begin_epoch - 1, model, ema)
    for epoch in range(begin_epoch, args.nepochs):

        logger.info('Current LR {}'.format(optimizer.param_groups[0]['lr']))

        train(epoch, begin_iter if epoch == begin_epoch else 0, model)
        lipschitz_constants.append(get_lipschitz_constants(model))
        logger.info('Lipsh: {}'.format(pretty_repr(lipschitz_constants[-1])))

        if args.learn_p:
            ords.append(get_ords(model))
            logger.info('Order: {}'.format(pretty_repr(ords[-1])))

        if args.act == 'LeakyLSwish':
            alpha, beta = get_activation_params(model)
            alphas.append(alpha)
            betas.append(beta)

            logger.info('alphas: {}'.format(pretty_repr(alphas[-1])))
            logger.info('betas: {}'.format(pretty_repr(betas[-1])))

        if args.learnable_concat:
            eta1, eta2, K1, K2 = get_learnable_params(model)
            concat_eta1.append(eta1)
            concat_eta2.append(eta2)
            concat_K1.append(K1)
            concat_K2.append(K2)

            logger.info('eta1: {}'.format(pretty_repr(concat_eta1[-1])))
            logger.info('eta2: {}'.format(pretty_repr(concat_eta2[-1])))
            logger.info('K1: {}'.format(pretty_repr(concat_K1[-1])))
            logger.info('K2: {}'.format(pretty_repr(concat_K2[-1])))

        if args.ema_val:
            test_bpd = validate(epoch, model, ema)
        else:
            test_bpd = validate(epoch, model)

        if args.scheduler and scheduler is not None:
            scheduler.step()

        if rank == 0:
            if test_bpd < best_test_bpd:
                best_test_bpd = test_bpd
                utils.save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'ema': ema,
                    'test_bpd': test_bpd,
                    'begin_epoch': epoch + 1,
                    'begin_iter': 0,
                }, os.path.join(args.save, 'models'), epoch, last_checkpoints, num_checkpoints=5)

            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                'test_bpd': test_bpd,
                'begin_epoch': epoch + 1,
                'begin_iter': 0,
            }, os.path.join(args.save, 'models', 'most_recent.pth'))

        if args.distributed:
            torch.distributed.barrier()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=str, default='cifar10', choices=[
            'mnist',
            'cifar10',
            'svhn',
            'celebahq',
            'celeba_5bit',
            'imagenet32',
            'imagenet64',
        ]
    )
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--imagesize', type=int, default=32)
    parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.

    parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

    parser.add_argument('--coeff', type=float, default=0.98)
    parser.add_argument('--vnorms', type=str, default='2222')
    parser.add_argument('--n-lipschitz-iters', type=int, default=None)
    parser.add_argument('--sn-tol', type=float, default=1e-3)
    parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

    parser.add_argument('--n-power-series', type=int, default=None)
    parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
    parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--n-exact-terms', type=int, default=2)
    parser.add_argument('--var-reduc-lr', type=float, default=0)
    parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
    parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

    parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='CLipSwish')
    parser.add_argument('--idim', type=int, default=512)
    parser.add_argument('--nblocks', type=str, default='16-16-16')
    parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
    parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
    parser.add_argument('--kernels', type=str, default='3-1-3')
    parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
    parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
    parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
    parser.add_argument('--fc-idim', type=int, default=128)
    parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
    parser.add_argument('--cdim', type=int, default=256)
    parser.add_argument('--clip_grad_norm', type=eval, choices=[True, False], default=True)

    # DenseNet
    parser.add_argument('--densenet', type=eval, choices=[True, False], default=True)
    parser.add_argument('--densenet_depth', type=int, default=3)
    parser.add_argument('--densenet_growth', type=int, default=172)
    parser.add_argument('--fc_densenet_growth', type=int, default=32)
    parser.add_argument('--learnable_concat', type=eval, choices=[True, False], default=True)
    parser.add_argument('--start_learnable_concat', type=int, default=25)
    parser.add_argument('--lip_coeff', help='Lipschitz coeff for DenseNet', type=float, default=0.98)

    # Monotone Flow
    parser.add_argument('--monotone_resolvent', type=eval, choices=[True, False], default=False)

    # Variational dequantization
    parser.add_argument('--var-deq', type=eval, choices=[True, False], default=False)
    parser.add_argument('--var-deq-nblocks', type=int, default=2)
    parser.add_argument('--var-deq-act', type=str, choices=ACT_FNS.keys(), default='CLipSwish')
    parser.add_argument('--var-deq-mf', type=eval, choices=[True, False], default=False)

    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
    parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
    parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=1000)
    parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--annealing-iters', type=int, default=0)
    parser.add_argument('--save', help='directory to save results', type=str, default='experiment1')
    parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
    parser.add_argument('--update-freq', type=int, default=1)

    parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
    parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
    parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
    parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-freq', help='Save model every so iterations (disabled if set to 0)', type=int, default=100)

    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
    parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)

    parser.add_argument('--distributed', type=eval, choices=[True, False], default=False)
    args = parser.parse_args()

    if args.save_freq > 0 and args.save_freq % args.update_freq != 0:
        raise ArgumentError('Save freq (%d) must be a multiple of update freq (%d)' % (args.save_freq, args.update_freq))

    # Random seed
    if args.seed is None:
        if not args.distributed:
            args.seed = np.random.randint(100000)
    elif args.distributed:
        raise ArgumentError('Cannot specify seed when using distributed training.')

    # Launch the main routine
    if args.distributed:
        assert torch.cuda.is_available()
        world_size = torch.cuda.device_count()
        assert world_size > 1

        if args.batchsize % world_size != 0:
            raise ArgumentError('Batch size must be a multiple of the world size (the number of GPUs)')
        args.batchsize = args.batchsize // world_size

        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, port, args))
    else:
        main_worker(0, 1, 0, args)
