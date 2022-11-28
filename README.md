# Invertible Monotone Operators for Normalizing Flows

Code for [Monotone Flows](https://arxiv.org/abs/2210.08176).

This work is inspired by [Invertible Residual Networks](https://arxiv.org/abs/1811.00995), [Residual Flows](https://arxiv.org/abs/1906.02735), [Implicit Normalizing Flows](https://arxiv.org/abs/2103.09527), and [i-DenseNets](https://arxiv.org/abs/2102.02694). The source is largely adapted from [i-DenseNets](https://github.com/yperugachidiaz/invertible_densenets) with fixed-point routines adapted from [Implicit Normalizing Flows](https://github.com/thu-ml/implicit-normalizing-flows/).

A BibTeX entry for LaTeX users:
```
@misc{ahn2022invertible,
      title={Invertible Monotone Operators for Normalizing Flows}, 
      author={Byeongkeun Ahn and Chiyoon Kim and Youngjoon Hong and Hyunwoo J. Kim},
      year={2022},
      booktitle={NeurIPS},
      eprint={2210.08176}
}
```


## Download datasets
* CIFAR10 is automatically downloaded. 
* The pre-processing steps and downloading of ImageNet32 are described in [Residual Flows](https://github.com/rtqichen/residual-flows).


## Setting up the environment
1. Create a new conda environment.
2. Install packages with the following command:
```
conda install pytorch=1.11.0 cudnn cudatoolkit=11.3 torchvision numpy pandas scikit-learn tqdm yacs matplotlib tabulate colorama ninja scipy termcolor -c pytorch -c nvidia -c conda-forge
```
3. Run the commands below.


## 1D toy experiments

* Experiment (a)
```
CUDA_VISIBLE_DEVICES=0 python train_1dexample.py --data multiplestepfunc --nblocks 2 --arch iresnet --save experiments/toy-1d/multiplestepfunc_rf_woLS --atol 1e-4 --rtol 1e-4 --densenet True --learnable_concat True --start_learnable_concat 0 --act CReLU --densenet_depth 4 --densenet_growth 128 --monotone_resolvent False --brute-force True --lr 0.01 --coeff 0.99 --lip_coeff 0.99 --batch_size 5000 --niters 15000 --weight-decay 0 --actnorm False
```

* Experiment (b)
```
CUDA_VISIBLE_DEVICES=0 python train_1dexample.py --data multiplestepfunc --nblocks 2 --arch iresnet --save experiments/toy-1d/multiplestepfunc_rf_wLS --atol 1e-4 --rtol 1e-4 --densenet True --learnable_concat True --start_learnable_concat 0 --act CReLU --densenet_depth 4 --densenet_growth 128 --monotone_resolvent False --brute-force True --lr 0.01 --coeff 0.99 --lip_coeff 0.99 --batch_size 5000 --niters 15000 --weight-decay 0 --actnorm True
```

* Experiment (c)
```
CUDA_VISIBLE_DEVICES=0 python train_1dexample.py --data multiplestepfunc --nblocks 1 --arch impflow --save experiments/toy-1d/multiplestepfunc_if_wLS --atol 1e-4 --rtol 1e-4 --densenet True --learnable_concat True --start_learnable_concat 0 --act CReLU --densenet_depth 4 --densenet_growth 128 --monotone_resolvent False --brute-force True --lr 0.01 --coeff 0.99 --lip_coeff 0.99 --batch_size 5000 --niters 15000 --weight-decay 0 --actnorm True
```

* Experiment (d)
```
CUDA_VISIBLE_DEVICES=0 python train_1dexample.py --data multiplestepfunc --nblocks 2 --save experiments/toy-1d/multiplestepfunc_mf_wLS --atol 1e-4 --rtol 1e-4 --densenet True --learnable_concat True --start_learnable_concat 0 --act CReLU --densenet_depth 4 --densenet_growth 128 --monotone_resolvent True --brute-force True --lr 0.01 --coeff 0.99 --lip_coeff 0.99 --batch_size 5000 --niters 15000 --weight-decay 0 --actnorm True
```


## 2D toy experiments

* 2 Spirals, i-DenseNets
```
python train_toy.py --data 2spirals --nblocks 10 --save experiments/toy-2d/2spirals_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* 2 Spirals, Monotone Flows
```
python train_toy.py --data 2spirals --nblocks 10 --save experiments/toy-2d/2spirals_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* 8 Gaussians, i-DenseNets
```
python train_toy.py --data 8gaussians --nblocks 10 --save experiments/toy-2d/8gaussians_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* 8 Gaussians, Monotone Flows
```
python train_toy.py --data 8gaussians --nblocks 10 --save experiments/toy-2d/8gaussians_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* Checkerboard, i-DenseNets
```
python train_toy.py --data checkerboard --nblocks 10 --save experiments/toy-2d/checkerboard_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* Checkerboard, Monotone Flows
```
python train_toy.py --data checkerboard --nblocks 10 --save experiments/toy-2d/checkerboard_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* Circles, i-DenseNets
```
python train_toy.py --data circles --nblocks 10 --save experiments/toy-2d/circles_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* Circles, Monotone Flows
```
python train_toy.py --data circles --nblocks 10 --save experiments/toy-2d/circles_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* Moons, i-DenseNets
```
python train_toy.py --data moons --nblocks 10 --save experiments/toy-2d/moons_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* Moons, Monotone Flows
```
python train_toy.py --data moons --nblocks 10 --save experiments/toy-2d/moons_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* Pinwheel, i-DenseNets
```
python train_toy.py --data pinwheel --nblocks 10 --save experiments/toy-2d/pinwheel_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* Pinwheel, Monotone Flows
```
python train_toy.py --data pinwheel --nblocks 10 --save experiments/toy-2d/pinwheel_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* Rings, i-DenseNets
```
python train_toy.py --data rings --nblocks 10 --save experiments/toy-2d/rings_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* Rings, Monotone Flows
```
python train_toy.py --data rings --nblocks 10 --save experiments/toy-2d/rings_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```

* Swissroll, i-DenseNets
```
python train_toy.py --data swissroll --nblocks 10 --save experiments/toy-2d/swissroll_idensenet --densenet True --learnable_concat True --start_learnable_concat 25000 --act CLipSwish --densenet_depth 3 --densenet_growth 16 --monotone_resolvent False --brute-force True
```

* Swissroll, Monotone Flows
```
python train_toy.py --data swissroll --nblocks 10 --save experiments/toy-2d/swissroll_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25000 --act CPila --densenet_depth 3 --densenet_growth 16 --monotone_resolvent True --brute-force True
```


## Image density modeling experiments

* MNIST (batch size 64, distributed, 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img.py --data mnist --nblocks 16-16-16 --save experiments/mnist --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 172 --monotone_resolvent True --n-exact-terms 10 --nworkers 0 --batchsize 64 --update-freq 4 --print-freq 20 --distributed False --mem-eff False --imagesize 28
```

* CIFAR-10 (batch size 64, distributed, 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10 --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 172 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 64 --update-freq 1 --print-freq 5 --distributed True
```

* ImageNet32 (batch size 256, distributed, 8 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_img.py --data imagenet32 --nblocks 32-32-32 --save experiments/imagenet32 --densenet True --learnable_concat True --start_learnable_concat 0 --act CPila --densenet_depth 3 --densenet_growth 172 --squeeze-first False --factor-out False --fc-end True --lr 0.004 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 256 --update-freq 1 --print-freq 5 --distributed True
```

* ImageNet64 (batch size 256, distributed, 8 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_img.py --data imagenet64 --nblocks 32-32-32 --save experiments/imagenet64 --densenet True --learnable_concat True --start_learnable_concat 0 --act CPila --densenet_depth 3 --densenet_growth 172 --squeeze-first True --factor-out True --fc-end True --lr 0.004 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 256 --update-freq 1 --print-freq 5 --distributed True
```


## Ablation experiments on CIFAR-10 image density modeling

* Row #1 (ablates both)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10 --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth 3 --densenet_growth 172 --monotone_resolvent False --n-exact-terms 10 --nworkers 4 --batchsize 64 --update-freq 1 --print-freq 5 --distributed True
```

* Row #2 (ablates the monotone formulation)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10 --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 172 --monotone_resolvent False --n-exact-terms 10 --nworkers 4 --batchsize 64 --update-freq 1 --print-freq 5 --distributed True
```

* Row #3 (ablates CPila)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10 --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth 3 --densenet_growth 172 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 64 --update-freq 1 --print-freq 5 --distributed True
```

* Row #4 (the same as the command for the CIFAR-10 density modeling experiment)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10 --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 172 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 64 --update-freq 1 --print-freq 5 --distributed True
```


## Variational dequantization experiments on CIFAR-10 and ImageNet32

* CIFAR-10
```
python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10_vdq --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 172 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 64 --lr 0.001 --update-freq 1 --print-freq 1 --distributed True --val-batchsize 64 --var-deq True --var-deq-nblocks 3 --var-deq-act CPila --var-deq-mf False
```

* ImageNet32
```
python train_img.py --data imagenet32 --nblocks 32-32-32 --save experiments/imagenet32_vdq --densenet True --learnable_concat True --start_learnable_concat 2 --act CPila --densenet_depth 3 --densenet_growth 172 --squeeze-first False --factor-out False --fc-end False --lr 0.002 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 1 --distributed True --var-deq True --var-deq-nblocks 3 --var-deq-act CPila --var-deq-mf False --neumann-grad True --mem-eff True
```


## Classification experiments on CIFAR-10

* Tiny (k=1), i-DenseNets
```
CUDA_VISIBLE_DEVICES=0,1 python train_img.py --data cifar10 --nblocks 1-1-1 --save experiments/cifar10_cls_tiny_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth 3 --densenet_growth 80 --monotone_resolvent False --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 5 --distributed True --fc-end False --mem-eff False --lr 0.001 --task classification --clip_grad_norm False --nepochs 200
```

* Tiny (k=1), Monotone Flows
```
CUDA_VISIBLE_DEVICES=0,1 python train_img.py --data cifar10 --nblocks 1-1-1 --save experiments/cifar10_cls_tiny_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 80 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 5 --distributed True --fc-end False --mem-eff False --lr 0.001 --task classification --clip_grad_norm False --nepochs 200
```

* Small (k=4), i-DenseNets
```
CUDA_VISIBLE_DEVICES=0,1 python train_img.py --data cifar10 --nblocks 4-4-4 --save experiments/cifar10_cls_tiny_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth 3 --densenet_growth 80 --monotone_resolvent False --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 5 --distributed True --fc-end False --mem-eff False --lr 0.001 --task classification --clip_grad_norm False --nepochs 200
```

* Small (k=4), Monotone Flows
```
CUDA_VISIBLE_DEVICES=0,1 python train_img.py --data cifar10 --nblocks 4-4-4 --save experiments/cifar10_cls_tiny_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 80 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 5 --distributed True --fc-end False --mem-eff False --lr 0.001 --task classification --clip_grad_norm False --nepochs 200
```

* Large (k=16), i-DenseNets
```
CUDA_VISIBLE_DEVICES=0,1 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10_cls_tiny_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25 --act CLipSwish --densenet_depth 3 --densenet_growth 80 --monotone_resolvent False --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 5 --distributed True --fc-end False --mem-eff False --lr 0.001 --task classification --clip_grad_norm False --nepochs 200
```

* Large (k=16), Monotone Flows
```
CUDA_VISIBLE_DEVICES=0,1 python train_img.py --data cifar10 --nblocks 16-16-16 --save experiments/cifar10_cls_tiny_monotoneflow --densenet True --learnable_concat True --start_learnable_concat 25 --act CPila --densenet_depth 3 --densenet_growth 80 --monotone_resolvent True --n-exact-terms 10 --nworkers 4 --batchsize 128 --update-freq 1 --print-freq 5 --distributed True --fc-end False --mem-eff False --lr 0.001 --task classification --clip_grad_norm False --nepochs 200
```
