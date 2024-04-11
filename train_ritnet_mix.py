"""
Author: Viet Nguyen
Date: 2022-11-04
"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow
import torch
from torch.utils.data import DataLoader
import pandas as pd

from vietlib.model.loss import DannLossObject
from vietlib.train.trainer import RitNetTrainer, RitNetTrainerMix
from vietlib.model.segment import DenseMobileNet2D_2, DenseMobileNet2D, DenseNet2D
from vietlib.utils import check_vscode_interactive, mIoU_segment, mIoU_v2

from vietlib.utils.experiment import CVExperiment, Experiment_v2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_root_folder", default="logs/dda_pipeline/pop2")
parser.add_argument("-e", "--experiment_name", default="ritnet_mix")
parser.add_argument("--n_folds", type=int, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--n_real_limit", default=None, type=int)
parser.add_argument("--n_synth_limit", default=None, type=int)
parser.add_argument("--source_domain", default="/data/rit_eyes.h5")
parser.add_argument("--target_domain", default="/data/open_eds_real.h5") # this one must always be the real dataset
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--use_lr_scheduler", default=False, action="store_true", help="this is good for training but have to trade for speed so much, only enable when needed")

parser.add_argument("--use_GPU", default=True)
parser.add_argument("--image_height", default=640, type=int)
parser.add_argument("--image_width", default=400, type=int)
parser.add_argument("--n_classes", default=4, type=int)
parser.add_argument("--channel", default=1, type=int)
parser.add_argument("--valid_step", default=8, type=int)
parser.add_argument("--epochs", default=150, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--starting_hidden_neuron", default=32, type=int)
parser.add_argument("--visualization_step_ratio", default=0.05, type=float)


# parse args
if check_vscode_interactive():
  args = parser.parse_args([])
else:
  args = parser.parse_args()
n_folds = args.n_folds
fold = args.fold
E = Experiment_v2(args.log_root_folder, args.experiment_name, fold=args.fold)
E.on_experiment_start(args)

from vietlib.data.transforms import transform_sequence
from vietlib.data import DannDataset2, DannDataset3, CVDataset
from vietlib.utils import get_nparams
from vietlib.model.loss import SegLossObject

# Device setup
if E.config.use_GPU and torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")


###### MAIN #####

# Load model based on model name
model = DenseNet2D(channel_size=E.config.starting_hidden_neuron, dropout=True,prob=0.2)
model = model.to(device)

nparams = get_nparams(model)
print(nparams)

# Define optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=E.config.learning_rate)
scheduler = None
if args.use_lr_scheduler:
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10, min_lr=0.0001)

train_include_idx = None
val_include_idx = None

# Setup kfold
if fold is not None and n_folds is not None:
  n_folds = int(n_folds)
  fold = int(fold)
  train_include_idx = [i for i in range(n_folds)]
  del train_include_idx[fold]
  val_include_idx = [fold]

## Dataset generation
train_dataset_1 = CVDataset(
  E.config.target_domain,
  split="train",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=train_include_idx,
  data_point_limit=args.n_real_limit
)
train_loader_1 = DataLoader(train_dataset_1, batch_size=E.config.batch_size, drop_last=True)

## Dataset generation
train_dataset_2 = CVDataset(
  E.config.source_domain,
  split="train",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=train_include_idx,
  data_point_limit=args.n_synth_limit
)
train_loader_2 = DataLoader(train_dataset_2, batch_size=E.config.batch_size, drop_last=True)

val_dataset_1 = CVDataset(
  E.config.target_domain,
  split="validation",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx,
  data_point_limit=args.n_real_limit
)
val_loader_1 = DataLoader(val_dataset_1, batch_size=E.config.batch_size, drop_last=True)

val_dataset_2 = CVDataset(
  E.config.source_domain,
  split="validation",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx,
  data_point_limit=args.n_synth_limit
)
val_loader_2 = DataLoader(val_dataset_2, batch_size=E.config.batch_size, drop_last=True)

test_dataset = CVDataset(
  E.config.target_domain,
  split="validation",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx
)
test_loader = DataLoader(test_dataset, batch_size=E.config.batch_size, drop_last=True)


visualization_step = int((len(train_loader_1) + len(train_loader_2)) * E.config.visualization_step_ratio)

loss_function = SegLossObject(device)

# Training object
trainer = RitNetTrainerMix(
  E,
  E.config.experiment_name,
  model,
  E.config.batch_size,
  train_loader_1,
  train_loader_2,
  val_dataset_1,
  val_loader_1,
  val_loader_2,
  test_loader,
  optimizer,
  scheduler,
  loss_function,
  total_epochs=E.config.epochs,
  valid_step=E.config.valid_step,
  log_root_folder=E.config.log_root_folder,
  save_history=True,
  device=device,
  visualization_step=visualization_step
)

trainer.train()

total_epochs = args.epochs

iou_bgs = []
iou_scleras = []
iou_irises = []
iou_pupils = []

val_loader_1 = DataLoader(val_dataset_1, batch_size=1, drop_last=True) # it is important to set batch_size to 1 so we can call .item()
for i, batch in enumerate(val_loader_1):
  if i % 50 == 0:
    print(f"batch {i}")
  with torch.no_grad():
    img, label, _, _ = batch
    output_device = model(img.to(device))
    
  [iou_bg, iou_sclera, iou_iris, iou_pupil] = mIoU_segment(output_device.cpu(), label, n_unique_labels=4)
  iou_bg, iou_sclera, iou_iris, iou_pupil = iou_bg.item(), iou_sclera.item(), iou_iris.item(), iou_pupil.item()

  iou_bgs.append(iou_bg)
  iou_scleras.append(iou_sclera)
  iou_irises.append(iou_iris)
  iou_pupils.append(iou_pupil)

miou_bg = np.mean(iou_bgs)
miou_sclera = np.mean(iou_scleras)
miou_iris = np.mean(iou_irises)
miou_pupil = np.mean(iou_pupils)

# Append to df
exp_name = args.experiment_name
df = pd.DataFrame.from_dict({
  "exp_name": [exp_name],
  "mean_miou_bg": [miou_bg],
  "mean_miou_sclera": [miou_sclera],
  "mean_miou_iris": [miou_iris],
  "mean_miou_pupil": [miou_pupil],
})

csv_path = os.path.join(E.log_directory, "stats.csv")
df.to_csv(csv_path, index=False)


