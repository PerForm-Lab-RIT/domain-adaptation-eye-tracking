"""
Author: Viet Nguyen
Date: 2022-11-04
"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse

from vietlib.utils import check_vscode_interactive, get_nparams, mIoU_segment, mIoU_v2
from vietlib.train.trainer import RITnetTrainer_v2
from vietlib.utils.experiment import CVExperiment, Experiment_v2
from vietlib.model.segment import DenseMobileNet2D_2, DenseMobileNet2D, DenseNet2D
from vietlib.utils.image import show_img
from vietlib.data.transforms import transform_sequence
from vietlib.data import CVDataset


parser = argparse.ArgumentParser()

parser.add_argument("-l", "--log_root_folder", default="logs/dda_pipeline/pop") # pop2 for the model training after ritnet on openeds
parser.add_argument("-e", "--experiment_name", default="ritnet_7_raw")
parser.add_argument("--n_folds", type=int, default=10)
parser.add_argument("--fold", type=int, default=4)
parser.add_argument("--n_limit", default=None, type=int)
parser.add_argument("--dataset_path", default="/data/open_eds_real.h5")
parser.add_argument("--val_dataset_path", default="/data/open_eds_real.h5")
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--use_lr_scheduler", default=False, action="store_true", help="this is good for training but have to trade for speed so much, only enable when needed")

parser.add_argument("--use_GPU", default=True)
parser.add_argument("--image_height", default=640, type=int)
parser.add_argument("--image_width", default=400, type=int)
parser.add_argument("--n_classes", default=4, type=int)
parser.add_argument("--channel", default=1, type=int)
parser.add_argument("--valid_step", default=100, type=int)
parser.add_argument("--epochs", default=170, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--starting_hidden_neuron", default=32, type=int)
parser.add_argument("--visualization_step_ratio", default=0.05, type=float)


# parse args
if check_vscode_interactive():
  args = parser.parse_args([])
else:
  args = parser.parse_args()

E = Experiment_v2(args.log_root_folder, args.experiment_name, fold=args.fold)
E.on_experiment_start(args)


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

fold = args.fold
n_folds = args.n_folds

# Setup kfold
if fold is not None and n_folds is not None:
  n_folds = int(n_folds)
  fold = int(fold)
  train_include_idx = [i for i in range(n_folds)]
  del train_include_idx[fold]
  val_include_idx = [fold]

## Dataset generation
train_dataset = CVDataset(
  E.config.dataset_path,
  split="train", 
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=train_include_idx,
  data_point_limit=args.n_limit
)

train_loader = DataLoader(train_dataset, batch_size=E.config.batch_size, drop_last=True)

val_dataset = CVDataset(
  E.config.dataset_path, 
  split="validation", 
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx
)
val_loader = DataLoader(val_dataset, batch_size=E.config.batch_size, drop_last=True)

# Training object
trainer = RITnetTrainer_v2(
  E, 
  model, 
  train_loader=train_loader,
  val_dataset=val_dataset,
  val_loader=val_loader, 
  optimizer=optimizer, 
  scheduler=scheduler,
  device=device,
  visualization_step=100
)

print("Training ...")

# actual training hapenning
trainer.train(E.config.epochs)


# Export metric stats
total_epochs = args.epochs

iou_bgs = []
iou_scleras = []
iou_irises = []
iou_pupils = []

val_dataset = CVDataset(
  E.config.val_dataset_path, 
  split="validation", 
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx
)
val_loader = DataLoader(val_dataset, batch_size=1, drop_last=True) # it is important to set batch_size to 1 so we can call .item()
for i, batch in enumerate(val_loader):
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



