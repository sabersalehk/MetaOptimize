import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
import copy



def build_optimizer(net, args, writer):

    
    if args.optimizer == 'AdamW':
        return AdamW_optimizer(net, lr=args.alpha0, writer=writer)
    if args.optimizer == 'HF': 
        from HF import HF
        args_base = {name:getattr(args, name+'_base') for name in ['alg', 'normalizer_param', 'momentum_param', 'weight_decay', 'Lion_beta2'] if getattr(args, name+'_base') is not None}
        args_meta = {name:getattr(args, name+'_meta') for name in ['alg', 'normalizer_param', 'momentum_param', 'weight_decay', 'Lion_beta2'] if getattr(args, name+'_meta') is not None}
        args_meta['meta_stepsize'] = args.meta_stepsize
        return HF(net, stepsize_groups=args.stepsize_groups, alpha0=args.alpha0, args_base=args_base, args_meta=args_meta, gamma=args.gamma, writer=writer)
    
    print('Error: optimizer ', args.optimizer, '  is not supported')
    0/0


class CosineDecayWithWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            self.finished_warmup = True
            return [self.min_lr + (base_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps))) / 2 
                    for base_lr in self.base_lrs]



class AdamW_optimizer():
    def __init__(self, net, lr, data=None, writer=None):
        self.lr = lr
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)
        total_steps = 422000 # Define total training steps
        warmup_steps = 10000  # Define warmup steps
        self.scheduler = CosineDecayWithWarmupScheduler(self.optimizer, warmup_steps, total_steps)


    def step(self, net, loss):
        
        net.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


