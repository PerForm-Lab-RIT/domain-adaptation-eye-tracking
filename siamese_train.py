"""
Author: Viet Nguyen
Date: 2022-09-20
"""

from logging import Logger
import tensorflow
from typing import List
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import gc

import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

##### Workaround to setup global config ############
from vietlib.train.siamese import SiameseTrainer
from vietlib.utils import check_vscode_interactive
from vietlib.utils.experiment import Experiment_v2

from vietlib.data.transforms import transform_sequence

from vietlib.data import CVDataset

config_path = None 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_root_folder", default="logs/dda_pipeline/siamese")
parser.add_argument("-e", "--experiment_name", default="sia_4")

parser.add_argument("--use_GPU", default=True)
parser.add_argument("--image_height", default=640, type=int)
parser.add_argument("--image_width", default=400, type=int)
parser.add_argument("--n_classes", default=4, type=int)
parser.add_argument("--channel", default=1, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--valid_step", default=100, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--source_domain", default="/data/rit_eyes.h5")
parser.add_argument("--target_domain", default="/data/open_eds_real.h5")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.0005, type=float)

# parse args
if check_vscode_interactive():
  args = parser.parse_args([])
else:
  args = parser.parse_args()

E = Experiment_v2(args.log_root_folder, args.experiment_name)
E.on_experiment_start(args)

# Device setup
if E.config.use_GPU and torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")

import torch
import torch.nn as nn

def classify(model, 
            data_generator: iter,
             y: np.ndarray,
             threshold: float
             ):
    """Function to test the accuracy of the model.

    Args:
      data_generator: batch generator,
      y: Array of actual target.
      threshold: minimum distance to be considered the same
      model: The Siamese model.
    Returns:
       float: Accuracy of the model.
    """
    accuracy = 0
    true_positive = false_positive = true_negative = false_negative = 0
    batch_start = 0

    for batch_one, batch_two in data_generator:
        batch_size = len(batch_one)
        batch_stop = batch_start + batch_size

        if batch_stop >= len(y):
            break
        batch_labels = y[batch_start: batch_stop]
        vector_one, vector_two = model((batch_one, batch_two))
        batch_start = batch_stop

        for row in range(batch_size):
            similarity = np.dot(vector_one[row], vector_two[row].T)
            same_question = int(similarity > threshold)
            correct = same_question == batch_labels[row]
            if same_question:
                if correct:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if correct:
                    true_negative += 1
                else:
                    false_negative += 1
            accuracy += int(correct)
    return {"accuracy":accuracy/len(y),
            "true_positive" : true_positive,
            "true_negative" : true_negative,
            "false_positive" : false_positive,
            "false_negative" : false_negative}



###### MAIN #####

## Dataset generation
train_source_dataset = CVDataset(
  data_path=E.config.source_domain,
  split="train",
  transform=transform_sequence,
  shuffle=True
)
train_source_loader = DataLoader(
  train_source_dataset,
  batch_size=E.config.batch_size,
  shuffle=True,
  drop_last=True
)

val_source_dataset = CVDataset(
  data_path=E.config.source_domain,
  split="validation", 
  transform=transform_sequence,
  data_point_limit=2000,
  shuffle=True
)
val_source_loader = DataLoader(
  val_source_dataset,
  batch_size=E.config.batch_size,
  shuffle=True,
  drop_last=True
)

train_target_dataset = CVDataset(
  data_path=E.config.target_domain,
  split="train",
  transform=transform_sequence,
  shuffle=True
)
train_target_loader = DataLoader(
  train_target_dataset,
  batch_size=E.config.batch_size,
  shuffle=True,
  drop_last=True
)

val_target_dataset = CVDataset(
  data_path=E.config.target_domain,
  split="validation", 
  transform=transform_sequence,
  data_point_limit=2000,
  shuffle=True
)
val_target_loader = DataLoader(
  val_target_dataset,
  batch_size=E.config.batch_size,
  shuffle=True,
  drop_last=True
)


# Training object
trainer = SiameseTrainer(
  E,
  train_source_loader=train_source_loader,
  val_source_loader=val_source_loader,
  train_target_loader=train_target_loader,
  val_target_loader=val_target_loader,
  device=device
)

# actual training hapenning
trainer.train(E.config.epochs)


# export loss fig
def rolling(arr, window=1000):
  average_y = []
  for ind in range(len(arr) - window + 1):
    average_y.append(np.mean(arr[ind:ind+window]))
  return average_y

n_epochs = len(trainer.history.history)
losses = []
val_losses = []
for i in range (n_epochs):
  losses += trainer.history.history[i]["Loss"]
  val_losses += trainer.history.history[i]["Val Loss"]
losses = rolling(losses, window=100)
val_losses = rolling(val_losses, window=10)

fig = plt.figure(figsize=(10, 3))
plt.plot(losses, label="Loss")
plt.plot(np.linspace(0, len(losses) - 1, len(val_losses)), val_losses, label="Val Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(E.log_directory, "sia_losses_12_e_rolling_loss_100_vloss_10.png"))
plt.close()
