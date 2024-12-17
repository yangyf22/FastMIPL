#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import time
import logging
import numpy as np
import torch
import random
import argparse
 

def str2bool(v): 
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# training settings
parser = argparse.ArgumentParser(
    description='Fast Multi-Instance Partial-Label Learning')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R', help='weight decay')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
parser.add_argument('--exp_dir', type=str, default='./logs/', help='logfile path')
parser.add_argument('--index', type=str, default='index', help='index path')
parser.add_argument('--ds', type=str, default='MNIST_MIPL', help='MNIST_MIPL, FMNIST_MIPL, ...')
parser.add_argument('--ds_suffix', type=str, default='1', help='the specific type of the data set')
parser.add_argument('--bs', type=int, default=1, help='batch size for training')
parser.add_argument('--nr_fea', type=int, default=784, help='feature dimension of an instance')
parser.add_argument('--nr_class', type=int, default=5, help='classes of bag')
parser.add_argument('--nr_samples', type=int, default=8, help='number of sampling')
parser.add_argument('--nr_trial', type=int, default=1, help='number of trial')
parser.add_argument('--normalize', type=str2bool, default=False, help='normalize the dataset, True or False')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0005)')
parser.add_argument('--smoke_test', type=str2bool, default=False, help='smoke_test, True or False')
parser.add_argument('--debug', type=str2bool, default=False, metavar='Debug Flag', help='debug flagm True or False')

args = parser.parse_args()

out_dir = args.exp_dir
if not args.no_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    args.cuda = True
else:
    device = torch.device("cpu")
    args.cuda = False

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Declare the log file
LOG_FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ]%(levelname)s >>> %(message)s"

log_name = "{}_{}_{}_lr{}epoch{}nrS{}.log".format(
    args.ds, args.ds_suffix, 
    time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    "".join(str(args.lr).split(".")), args.epochs, args.nr_samples
    )

if args.debug:
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        stream=sys.stdout
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(out_dir, log_name),
        filemode="w",
        format=LOG_FORMAT
    )

logging.info("Record the params {}\n".format(args))


def seed_everything(seed):
    random.seed(seed)                                       # Python random module
    np.random.seed(seed)                                    # Numpy module
    torch.manual_seed(seed)                                 # Current CPU
    torch.cuda.manual_seed(seed)                            # Current GPU
    torch.cuda.manual_seed_all(seed)                        # All GPU (Optional)
    torch.backends.cudnn.benchmark = False                  # Close optimization
    torch.backends.cudnn.deterministic = True               # Close optimization
    os.environ['PYTHONHASHSEED'] = str(seed)                # Python hash seed
    torch.use_deterministic_algorithms(True)                # Enable deterministic algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'         # Set CUBLAS workspace

