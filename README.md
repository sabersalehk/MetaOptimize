# MetaOptimize

MetaOptimize is a framework that can wrap around any first-order optimization algorithm, tuning step sizes on the fly to minimize a specific form of regret that accounts for long-term effect of step sizes on training, through a discounted sum of future losses.

# Citation

This repository contains an implementation of the MetaOptimize in the following paper:

```bibtex
@article{sharifnassab2024metaoptimize,
  title={Metaoptimize: A framework for optimizing step sizes and other meta-parameters},
  author={Sharifnassab, Arsalan and Salehkaleybar, Saber and Sutton, Richard},
  journal={International Conference on Machine Learning (ICML)},
  url={https://arxiv.org/pdf/2402.02342},
  year={2025}
}
```

# Install 

pip3 install git+https://github.com/sabersalehk/MetaOptimize.git 

## Example: Train an MLP on MNIST with MetaOptimize

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from meta_optimize.optimizer import MetaOptimize_AdamW_Lion_hg

# Define MLP (Net_2layer_M1)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load MNIST
def get_dataloaders(batch_size=64):
    transform = transforms.ToTensor()
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader, testloader = get_dataloaders()
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = MetaOptimize_AdamW_Lion_hg(
    model.parameters(),
    meta_stepsize=1e-3,
    alpha0=1e-4,
    stepsize_blocks='scalar',
    gamma=1.0,
    b1=0.9,
    b2=0.999,
    meta_b1=0.99,
    meta_c1=0.9,
    weight_decay=0.1
)

# Training loop
for epoch in range(5):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss={total_loss / total:.4f}, Acc={acc:.2f}%")



# Codes for Experiments in the Paper

The code for the experiments in the paper is available in the `code` folder.

- Experiments on **supervised learning tasks** (CIFAR-10, ImageNet, and TinyStories) are provided in the `Supervised_task.zip` file.
- Experiments on **continual learning** (CIFAR-100) are available in the `Continual_CIFAR100_experiments.zip` file.


## About Experiments on Supervised Learning Tasks

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



