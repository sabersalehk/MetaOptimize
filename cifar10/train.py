#import psutil; process = psutil.Process()  # for memory monitoring
import time
start_time = time.time()
import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from build_network import build_network
from Optimizers.build_optimizer import build_optimizer
from load_data import load_data, compute_test_accuracy


def parse_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--save-directory", type=str, default='',
                        help="whether to save the outputs of tensorboard")
    parser.add_argument("--run-name", type=str, default=None,
                        help="name of the current run")
    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        help="name of the current run")
    parser.add_argument("--NN-name", type=str, default='ResNet18', # M1 is a 2-layer small network, M2 is a 3-layer network of width 512 for MNIST
                        help="name of Neural Network")
    parser.add_argument("--verbos", type=int, default=1,
                        help="how much printing. 0:minimal printing,  1:moderate_printing")
    parser.add_argument("--max-time", type=str, default="24:00:00",
                        help="maximum runtime of the program in hh:mm:ss format")
    
    
    # Trainig
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="name of the base and meta optimizer (if we have a meta optimizer)")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="nameber if epochs")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="nameber if epochs")
    
    
    # Hyperparameters for SGD, RMSProp, Adam, AdamW, ...
    parser.add_argument("--base-lr", type=float, default=1e-2,
                        help="learning rage of the base algorithm")
    parser.add_argument("--base-momentum", type=float, default=0.9,
                        help="momentum weight of the base algorithm")
    
    
    # Arguemnts for Meta algorithms
    parser.add_argument("--gamma", type=float, default=.9999,
                        help="decay factor")
    parser.add_argument("--lambda-of-meta-rmsp", type=float, default=.9999,
                        help="meta rmsprop lambda")
    parser.add_argument("--mask-blocks", type=str, default=None,
                        help="determines the blocks of G matrix to be mask")
    
    
    # Arguemnts for Base and Meta algorithms
    parser.add_argument("--alg-base", type=str, default=None)
    parser.add_argument("--normalizer-param-base", type=float, default=None)
    parser.add_argument("--momentum-param-base", type=float, default=None)
    parser.add_argument("--weight-decay-base", type=float, default=None)
    parser.add_argument("--Lion-beta2-base", type=float, default=None)

    parser.add_argument("--alg-meta", type=str, default=None)
    parser.add_argument("--normalizer-param-meta", type=float, default=None)
    parser.add_argument("--momentum-param-meta", type=float, default=None)
    parser.add_argument("--weight-decay-meta", type=float, default=None)
    parser.add_argument("--Lion-beta2-meta", type=float, default=None)

    parser.add_argument("--stepsize-groups", type=str, default=None)



    parser.add_argument("--meta-stepsize", type=float, default=None)
    parser.add_argument("--alpha0", type=float, default=1e-4)

    args = parser.parse_args()
    return args





##################
####    Main
##################

if __name__ == "__main__":

    # load arguments
    args = parse_args()
    if args.normalizer_param_base == -1:
        args.normalizer_param_base = None
    if args.momentum_param_base == -1:
        args.momentum_param_base = None
    if args.weight_decay_base == -1:
        args.weight_decay_base = None
    if args.Lion_beta2_base == -1:
        args.Lion_beta2_base = None
    if args.normalizer_param_meta == -1:
        args.normalizer_param_meta = None
    if args.momentum_param_meta == -1:
        args.momentum_param_meta = None
    if args.weight_decay_meta == -1:
        args.weight_decay_meta = None
    if args.Lion_beta2_meta == -1:
        args.Lion_beta2_meta = None
    if args.meta_stepsize == -1:
        args.meta_stepsize = None
    if args.meta_stepsize == -1:
        args.meta_stepsize = None


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_time_in_seconds = sum(int(args.max_time.split(':')[i])*3600/(60**i) for i in range(3))
    run_name = args.run_name if args.run_name is not None else f"{args.dataset}__{args.optimizer}__{args.seed}__{int(time.time())}"

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # For plotting
    writer = SummaryWriter(os.path.join(args.save_directory, 'Tensorboard_outputs', run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)

    # build NN
    #print("=> creating model '{}'".format(args.arch))
   
    device = args.device
    
    #net = models.__dict__[args.NN_name]().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    net = build_network(args.NN_name, args.device)

    # build optimizer
    optimizer = build_optimizer(net, args, writer)

    # load data
    trainloader, testloader = load_data(args.dataset, args.batch_size, args.seed)
    
    

    # Trainig loop
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        epoch_time = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(args.device), data[1].to(args.device) #data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            if True: # update performance metrics for plotting
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            optimizer.step(net, loss)

        
        ## Plot performance
        train_loss = running_loss / total_predictions
        train_accuracy = 100 * correct_predictions / total_predictions
        test_accuracy = compute_test_accuracy(net, testloader, args.device)
        
        writer.add_scalar("Performance/train_loss", train_loss, epoch)
        writer.add_scalar("Performance/train_accuracy", train_accuracy, epoch)
        writer.add_scalar("Performance/test_accuracy", test_accuracy, epoch)
        #writer.add_scalar("Memory", int(process.memory_info().rss/ (1024 ** 2)), epoch) # For memory monitoring

        if args.verbos: print('Epoch %d, Train Accuracy: %.2f %%, Test Accuracy: %.2f %%' % (epoch, train_accuracy, test_accuracy))
        
        ## termination condition based on time limit
        if time.time()-start_time > max_time_in_seconds-1.2*(time.time()-epoch_time) - 60: break


writer.close()
print(int((time.time()-start_time)/60),' minutes')

