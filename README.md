# MetaOptimize

MetaOptimize is a framework that can wrap around any first-order optimization algorithm, tuning step sizes on the fly to minimize a specific form of regret that accounts for long-term effect of step sizes on training, through a discounted sum of future losses.

# Citation

This repository contains an implementation of the MetaOptimize of the following paper:

```bibtex
@article{sharifnassab2024metaoptimize,
  title={Metaoptimize: A framework for optimizing step sizes and other meta-parameters},
  author={Sharifnassab, Arsalan and Salehkaleybar, Saber and Sutton, Richard},
  journal={International Conference on Machine Learning (ICML)},
  url={https://arxiv.org/pdf/2402.02342},
  year={2025}
}
```

# Usage

Three folders contain codes for the experiments on CIFAR10, ImageNet, and TinyStories datasets. The main file to run the code is main.py in each folder. As an example, to run a (base, meta) = (AdamW, Lion) combination, the following command can be used:

python3 train.py --optimizer HF --alg-base AdamW --weight-decay-base .1 --normalizer-param-base .999 --momentum-param-base .9 --Lion-beta2-base -1 --alg-meta Lion --meta-stepsize 1e-3 --alpha0 1e-6 --stepsize-groups scalar --weight-decay-meta 0 --normalizer-param-meta -1 --momentum-param-meta .99 --Lion-beta2-meta .9 --seed 0 --gamma 1 --run-name 1 --save-directory outputs --max-time 00:05:00

If you want to run the code with the default configuration, use the following command:

python3 train.py


The set of arguments are:
* optimizer: currently HF or AdamW (only in ImageNet or TinyStories dataset)
* alg-base: base update (the options are: Adam, Lion, RMSProp, SGDm (SGD with momentum))
* alg-meta: meta update (the options are: Adam, Lion)
* weight-decay-base: $\kappa$
* normalizer-param-base: $\lambda$
* momentum-param-base: $\rho$
* Lion-beta2-base : $c$
* weight-decay-meta: $\bar{\kappa}$
* normalizer-param-meta: $\bar{\lambda}$
* momentum-param-meta: $\bar{\rho}$
* Lion-beta2-meta : $\bar{c}$
* meta-stepsize : $\eta$
* alpha0 : $\alpha_0$
* gamma : $\gamma$ 
* seed: the seed number
* run-name: the name of run
* save-directory: the location of saving the outputs
* max-time: maximum allowed time to run the algorithm

In ImageNet dataset, we used the implementation of https://pytorch.org/examples/ to read the data. Please update the argument 'data' in this file to the path of your ImageNet dataset. For TinyStories, we used the implementation of code in https://github.com/karpathy/llama2.c to read the data and tokenize it. Please follow the instructions there to tokenize the data. Moreover, please set DATA_CACHE_DIR in tinystories.py to the path of tokenized data.

# Requirements

The required packages are:
* numpy==1.23.5
* pytest==7.4.0
* Requests==2.31.0
* sentencepiece==0.1.99
* torch==2.0.1
* torchvision==0.15.2
* tqdm==4.64.1
* wandb==0.15.5



