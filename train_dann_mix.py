"""
Author: Viet Nguyen
Date: 2022-11-04
"""

from logging import Logger
import tensorflow
from typing import List
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from vietlib.train.trainer import DannTrainer
from vietlib.utils import check_vscode_interactive, free_memories, mIoU_segment, time_record
from vietlib.utils.experiment import PROFILE_NAME, DannProfile, Experiment_v2, Experiment
from vietlib.model.segment import DannOriginal
from vietlib.data.transforms import transform_sequence
from vietlib.utils import mIoU_v2, get_nparams, get_predictions
from vietlib.data import CVDataset
from vietlib.model.loss import SegLossObject

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("--config_path")
parser.add_argument("-l", "--log_root_folder", default="logs/dda_pipeline/pop2")
parser.add_argument("-e", "--experiment_name", default="dann_mix")
parser.add_argument("--n_folds", type=int, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--n_real_limit", default=None, type=int)
parser.add_argument("--n_synth_limit", default=None, type=int)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--source_domain", default="/data/rit_eyes.h5")
parser.add_argument("--target_domain", default="/data/open_eds_real.h5") # this one must always be the real dataset
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

# Device setup
if E.config.use_GPU and torch.cuda.is_available():
  device=torch.device("cuda")
else:
  device=torch.device("cpu")

model = DannOriginal(channel_size=E.config.starting_hidden_neuron, dropout=True,prob=0.2)
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
train_source_dataset = CVDataset(
  data_path=E.config.source_domain,
  split="train",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=train_include_idx,
  data_point_limit=args.n_synth_limit
)
train_source_loader = DataLoader(train_source_dataset, batch_size=E.config.batch_size, drop_last=True)

val_source_dataset = CVDataset(
  data_path=E.config.source_domain,
  split="train", 
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx
)
val_source_loader = DataLoader(val_source_dataset, batch_size=E.config.batch_size, drop_last=True)

train_target_dataset = CVDataset(
  data_path=E.config.target_domain,
  split="train",
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=train_include_idx,
  data_point_limit=args.n_real_limit
)
train_target_loader = DataLoader(train_target_dataset, batch_size=E.config.batch_size, drop_last=True)

val_target_dataset = CVDataset(
  data_path=E.config.target_domain,
  split="train", 
  transform=transform_sequence,
  n_segments=n_folds,
  include_idx=val_include_idx,
  data_point_limit=args.n_real_limit # put here so that we do not have to validate the whole target dataset each epoch for like 4000 epochs
)
val_target_loader = DataLoader(val_target_dataset, batch_size=E.config.batch_size, drop_last=True)

visualization_step = 500

class DannTrainer3():
  def __init__(self,
      experiment: Experiment,
      model,
      train_source_loader=None,
      val_source_loader=None,
      train_target_loader=None,
      val_target_loader=None,
      val_dataset_target=None,
      optimizer=None,
      scheduler=None,
      device=torch.device("cpu"),
      visualization_step: int=0) -> None:

    # Initialize variables
    self.E = experiment
    self.model = model
    self.batch_size = self.E.config.batch_size
    self.train_target_loader = train_target_loader
    self.val_target_loader = val_target_loader
    self.train_source_loader = train_source_loader
    self.val_source_loader = val_source_loader
    self.optimizer = optimizer
    self.valid_step = self.E.config.valid_step
    self.device = device
    self.val_dataset_target = val_dataset_target
    self.scheduler = scheduler
    if visualization_step != 0:
      self.visualization_step = visualization_step

    # Additional setup
    self.val_target_iter = iter(val_target_loader)
    self.val_source_iter = iter(val_source_loader)
    self.domain_loss_obj = torch.nn.BCELoss()
    self.seg_loss_obj = SegLossObject(device)

    self.prev_val_loss = 1e8

    # Construct model path
    self.model_path = os.path.join(self.E.log_directory, "checkpoints", self.E.config.experiment_name + ".pth")
    # Path for saving training results in numpy formats
    self.history_path = os.path.join(self.E.log_directory, "training_history.npy")
    # Initialize model weights
    if self.model_path != None and os.path.exists(self.model_path):
      try:
        self.model.load_state_dict(torch.load(self.model_path))
        self.E.logger.write(f"Model weights loaded!")
      except:
        self.E.logger.write(f"Model weights not exist! Training new model...")
    for p in self.model.parameters():
      p.requires_grad = True
    # History profile
    self.history = DannProfile(self.history_path)
    self.current_epoch = self.history.get_current_epoch()

  def _export_batch_prediction(self, batch_img, epoch, step, is_target_domain=None, name="seg"):
    fig = plt.figure(figsize=(20, 35))

    subfigs = fig.subfigures(nrows=self.batch_size, ncols=1)

    with torch.no_grad():
      data = batch_img.to(self.device)
      output, domain_out_device, latent, el_params, flattened = self.model(data, self.alpha[epoch], self.backward_alpha[epoch])
      predict = get_predictions(output)

      for j, subfig in enumerate(subfigs):

        pred_img_ori = predict[j].cpu().numpy()
        pred_img = pred_img_ori/3.0
        inp = batch_img[j].squeeze() * 0.5 + 0.5
        img_orig = np.clip(inp,0,1)
        img_orig = np.array(img_orig)

        # Set title for the whole row
        pred_dom_val = "%.3f"%(domain_out_device[j,0].item())

        if is_target_domain != None:
          subfig.suptitle(f"Label {self.E.config.domain_class[is_target_domain[j,0].int().item()]} domain. Predict target domain prob: {pred_dom_val}")
        
        # create 1x5 subplots per subfig
        axes = subfig.subplots(nrows=1, ncols=5)

        # axes[j // 2, j % 2].figure(facecolor='white')
        axes[0].axis("off")
        axes[0].imshow(img_orig, cmap='gray')
        axes[0].grid(False)
        
        axes[1].axis("off")
        axes[1].imshow(pred_img)
        axes[1].grid(False)

        axes[2].axis("off")
        axes[2].imshow(pred_img_ori == 1)
        axes[2].grid(False)

        axes[3].axis("off")
        axes[3].imshow(pred_img_ori == 2)
        axes[3].grid(False)

        axes[4].axis("off")
        axes[4].imshow(pred_img_ori == 3)
        axes[4].grid(False)

    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(self.E.figure_folder, f"{name}_epoch_{epoch}_step_{step}"))
    plt.close(fig)

    del fig, j, subfig, predict, output, domain_out_device, \
      latent, el_params, flattened, data, subfigs

  @time_record
  def _train_one_epoch(self, epoch) -> List[float]:
    # Max instead of min is good because we will be able to iterate through less-data-dataset multiple time per epoch
    len_dataloader = max(len(self.train_source_loader), len(self.train_target_loader))
    train_source_it = iter(train_source_loader)
    train_target_it = iter(train_target_loader)

    self.E.logger.write_silent(f"---------- Training epoch {epoch} on source data (label classfier) ----------")
    
    for step in range(len_dataloader):
      
      ###### one step using source data: train the segmentation network and the domain classifier
      try:
        img_s, label_s, spatial_weights_s, distance_map_s = next(train_source_it)
      except:
        train_source_it = iter(train_source_loader)
        img_s, label_s, spatial_weights_s, distance_map_s = next(train_source_it)

      # Prepare copying tensor to designated cpu/gpu. Model inference
      img_device_s = img_s.to(self.device)
      output_device_s, domain_out_device_s, latent_s, el_params_s, flattened_s = self.model(img_device_s, self.alpha[epoch], self.backward_alpha[epoch])
      label_device_s = label_s.to(self.device)
      domain_source_label = torch.zeros((self.batch_size, 1)).float().to(self.device)
      
      # Compute loss
      loss_source_seg = self.seg_loss_obj(label_device_s, output_device_s, spatial_weights_s, distance_map_s, self.alpha[epoch])
      loss_source_domain = self.domain_loss_obj(domain_out_device_s, domain_source_label)

      ###### one step using target data: train only the domain classifier
      try:
        img_t, label_t, spatial_weights_t, distance_map_t = next(train_target_it)
      except:
        train_target_it = iter(train_target_loader)
        img_t, label_t, spatial_weights_t, distance_map_t = next(train_target_it)

      # Prepare copying tensor to designated cpu/gpu. Model inference
      img_device_t = img_t.to(self.device)
      output_device_t, domain_out_device_t, latent_t, el_params_t, flattened_t = self.model(img_device_t, self.alpha[epoch], self.backward_alpha[epoch])
      domain_target_label = torch.ones((self.batch_size, 1)).float().to(self.device) # bug here
      label_device_t = label_t.to(self.device)

      # Loss
      loss_target_seg = self.seg_loss_obj(label_device_t, output_device_t, spatial_weights_t, distance_map_t, self.alpha[epoch])
      loss_target_domain = self.domain_loss_obj(domain_out_device_t, domain_target_label)

      # Backward loss # Train dann mix
      loss = loss_source_seg + loss_source_domain + loss_target_seg + loss_target_domain
      # Zero your gradients for every batch!
      self.optimizer.zero_grad()
      # Backward loss
      loss.backward()
      # Adjust learning weights
      self.optimizer.step()

      # Append loss
      self.history.add_element(loss_source_seg.squeeze().item(), PROFILE_NAME.ssl, epoch)
      self.history.add_element(loss_source_domain.squeeze().item(), PROFILE_NAME.sdl, epoch)
      self.history.add_element(loss_target_domain.squeeze().item(), PROFILE_NAME.tdl, epoch)
      self.history.add_element(loss.squeeze().item(), PROFILE_NAME.l, epoch)

      # Append gradient for seg
      bf: torch.Tensor = torch.clone(self.model.down_block5.bn.weight.grad).cpu().detach()
      l2bf: float = np.linalg.norm(bf.numpy()) # shape (B, features) -> float
      self.history.add_element(l2bf, PROFILE_NAME.l2bf_s, epoch)

      # Compute mean iou and add it to the currnet epoch stats
      mean_iou: float = mIoU_v2(output_device_t.cpu(), label_t)
      self.history.add_element(mean_iou, PROFILE_NAME.miou_on_target, epoch)
      
      # Validation and logging work per validation steps
      if step % self.valid_step == 0:
        try:
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_target_iter)
        except:
          self.val_target_iter = iter(self.val_target_loader)
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_target_iter)
        
        with torch.no_grad():
          # Prepare copying tensor to designated cpu/gpu. Model inference
          val_img_device = val_img.to(self.device)
          val_output_device, val_domain_out_device, val_latent, val_el_params, val_flattened = self.model(val_img_device, self.alpha[epoch], self.backward_alpha[epoch])
          val_label_device = val_label.to(self.device)

        # Compute mean iou and add it to the currnet epoch stats
        mean_iou: float = mIoU_v2(val_output_device.cpu(), val_label)
        self.history.add_element(mean_iou, PROFILE_NAME.v_miou_on_target, epoch)

        # Log every valid step
        self.history.log_latest_data(self.E.logger, step, len_dataloader, silent=True)

        del val_domain_out_device, val_latent, val_el_params, \
          val_flattened, val_output_device, val_label_device, \
          mean_iou, val_img_device, val_img, val_label, val_spatial_weights, \
          val_distance_map

      if step == 0 and epoch % 10 == 0:
        try:
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_target_iter)
        except:
          self.val_target_iter = iter(self.val_target_loader)
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_target_iter)
        self._export_batch_prediction(val_img, epoch, step, name="seg")
        del val_img, val_label, val_spatial_weights, val_distance_map

  def validate_dataset(self, epoch):

    losses = []

    for i, batch in enumerate(self.val_target_loader):
      if i % 100 == 0:
        print(f"Validate batch {i}")
        
      val_img, val_label, val_spatial_weights, val_distance_map = batch

      with torch.no_grad():
        # Prepare copying tensor to designated cpu/gpu. Model inference
        val_img_device = val_img.to(self.device)
        val_output_device, val_domain_out_device, val_latent, val_el_params, val_flattened = self.model(val_img_device, self.alpha[epoch], self.backward_alpha[epoch])

        domain_target_label = torch.ones((self.batch_size, 1)).float().to(self.device)
        loss_target_domain = self.domain_loss_obj(val_domain_out_device, domain_target_label)

      losses.append(loss_target_domain.item()) # HKS: add the mean loss of the batch instead of a single loss to the array

    return np.mean(losses)

  def train(self, total_epochs) -> List[np.ndarray]:
    # Setup alpha
    self.alpha = np.zeros(((total_epochs)))
    self.alpha[0: np.min([125, total_epochs])] = 1 - np.arange(1, np.min([125, total_epochs]) + 1) / np.min([125, total_epochs])
    if total_epochs > 125:
      self.alpha[125:] = 1

    # Setup backward alpha
    self.backward_alpha = np.zeros(((total_epochs)))
    self.backward_alpha[0: np.min([10, total_epochs])] = 1 - np.arange(1, np.min([10, total_epochs]) + 1) / np.min([10, total_epochs])
    if total_epochs > 10:
      self.backward_alpha[10:] = 1

    while self.current_epoch < total_epochs:
      # Train one epoch
      self._train_one_epoch(self.current_epoch) # Train source (synthetic) dataset on feature extractors and label classifier

      # VALIDATE
      if self.scheduler is not None:
        val_loss = self.validate_dataset(self.current_epoch)
        self.scheduler.step(val_loss)
        self.E.logger.write_silent(f"val_loss: {val_loss}, current lr after scheduler step: {self.scheduler.optimizer.param_groups[0]['lr']}")

        if val_loss < self.prev_val_loss:
          self.E.logger.write_silent(f"Best model at epoch {self.current_epoch}")
          torch.save(self.model.state_dict(), os.path.join(self.E.log_directory, "checkpoints", f"best_model.pth"))
          self.prev_val_loss = val_loss

      # Save data
      self.history.save_numpy_data()
      torch.save(self.model.state_dict(), self.model_path)

      # Save a separate model every 5 epoch
      if self.current_epoch % 50 == 0:
        tmp_path = os.path.join(self.E.log_directory, "checkpoints", f"{self.E.config.experiment_name}_epoch_{self.current_epoch}.pth")
        torch.save(self.model.state_dict(), tmp_path)

      # Increment current epoch
      self.current_epoch += 1

# Training object
trainer = DannTrainer3(
  E,
  model,
  train_source_loader,
  val_source_loader,
  train_target_loader,
  val_target_loader,
  val_target_dataset,
  optimizer,
  scheduler,
  device=device,
  visualization_step=visualization_step
)

# actual training hapenning
trainer.train(E.config.epochs)

# Export metric stats
total_epochs = args.epochs

alpha = np.zeros(((total_epochs)))
alpha[0: np.min([125, total_epochs])] = 1 - np.arange(1, np.min([125, total_epochs]) + 1) / np.min([125, total_epochs])
if total_epochs > 125:
  alpha[125:] = 1

# Setup backward alpha
backward_alpha = np.zeros(((total_epochs)))
backward_alpha[0: np.min([10, total_epochs])] = 1 - np.arange(1, np.min([10, total_epochs]) + 1) / np.min([10, total_epochs])
if total_epochs > 10:
  backward_alpha[10:] = 1


total_epochs = args.epochs

iou_bgs = []
iou_scleras = []
iou_irises = []
iou_pupils = []

val_target_loader = DataLoader(val_target_dataset, batch_size=1, drop_last=True) # it is important to set batch_size to 1 so we can call .item()
for i, batch in enumerate(val_target_loader):
  if i % 50 == 0:
    print(f"batch {i}")
  with torch.no_grad():
    img, label, _, _ = batch
    output_device, _, _, _, _ = model(img.to(device), alpha[total_epochs - 1], backward_alpha[total_epochs - 1])
    
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


