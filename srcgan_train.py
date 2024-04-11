"""
Author: Viet Nguyen
Date: 2022-09-18
"""

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow
import warnings

import sys

import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

import h5py

from vietlib.data import PairedImageDatasetWithLabel
from vietlib.data.transforms import MaskToTensor
from vietlib.model.cgan import Discriminator, GeneratorResNet
from vietlib.model.loss import StructureRetainLoss
from vietlib.model.segment import SobelMultiChannel
from vietlib.model.cnn import ResidualBlock
from vietlib.train import Trainer, save_experiment
from vietlib.utils import check_vscode_interactive
from vietlib.utils.experiment import Experiment_v2

from vietlib.utils.experiment import PROFILE_NAME, Experiment, MLExperiment
from vietlib.utils.image import one_hot_label, show_img
import argparse

from vietlib.train.cgan import CGANTrainer_v2
import torchvision.transforms as transforms

transforms_ = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])

# Create experiment with argeparse

parser = argparse.ArgumentParser()
# Core variables
parser.add_argument("-l", "--log_root_folder", default="../../../logs/dda_pipeline/srcgan")
parser.add_argument("-e", "--experiment_name", default="srcgan_test")
parser.add_argument("-f", "--fold", default=None)

# Training variables
parser.add_argument("--source_domain", default="/data/rit_eyes.h5") # rc folder
parser.add_argument("--target_domain", default="/data/open_eds_real.h5") # rc folder
parser.add_argument("--use_GPU", default=True)
parser.add_argument("--image_height", default=640, type=int)
parser.add_argument("--image_width", default=400, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--channel", default=1, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--n_residual_blocks", default=8, type=int)
parser.add_argument("--n_classes", default=4, type=int)
parser.add_argument("--use_segment", default=False)
parser.add_argument("--valid_step", default=50, type=int)

# CGAN loss
parser.add_argument("--id_loss", default=10, type=float)
parser.add_argument("--cycle_loss", default=10, type=float)
parser.add_argument("--gan_loss", default=1, type=float)

# Sr loss
parser.add_argument("--gamma_edge", default=0, type=float)
parser.add_argument("--gamma_var", default=0, type=float)
parser.add_argument("--gamma_mean", default=0, type=float)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--beta", default=0.5, type=float)
parser.add_argument("--edge_threshold", default=0.3, type=float)

parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--starting_hidden_neuron", default=32, type=int)



# parse args
if check_vscode_interactive():
  args = parser.parse_args([])
else:
  args = parser.parse_args()

# Device setup
if args.use_GPU and torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")

# Load experiment
E = Experiment_v2(args.log_root_folder, args.experiment_name, fold=args.fold)
E.on_experiment_start(args)

# data (img)
img_height = E.config.image_height
img_width = E.config.image_width
channels = E.config.channel
input_shape = (channels, img_height, img_width) 

# training
n_epochs = E.config.epochs # number of epochs of training
batch_size = E.config.batch_size # size of the batches
b1 = 0.5 # adam : decay of first order momentum of gradient
b2 = 0.999 # adam : decay of first order momentum of gradient

n_residual_blocks = E.config.n_residual_blocks

G_AB = GeneratorResNet(input_shape, E.config.n_classes, n_residual_blocks, use_segment=E.config.use_segment)
G_BA = GeneratorResNet(input_shape, E.config.n_classes, n_residual_blocks, use_segment=E.config.use_segment)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

summary(G_AB.to(device), input_size=(1, 640, 400))

sobel_img = SobelMultiChannel(1, normalize=True).to(device)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)


cgan = CGANTrainer_v2(
  E,
  G_AB, G_BA, D_A, D_B,
  device,
  visualization_step=E.config.valid_step
)

cgan.train(E.config.epochs)

E.on_experiment_end()
