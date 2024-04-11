
import torch
import torch.nn as nn

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

from ..utils.experiment import Experiment
from ..model.siamese import ContrastiveLoss, SiameseNet2D
from ..utils import get_nparams, time_record
from ..utils.experiment import PROFILE_NAME, DannProfile, MLExperiment, Profile
from ..model.segment import FeatureExtractor

from ..data import DannDataset3, CVDataset, DannDataset3Short
from ..model.loss import DannLossObject, SegLossObject


class SiameseTrainer():
  def __init__(self,
      experiment: Experiment,
      train_source_loader=None,
      val_source_loader=None,
      train_target_loader=None,
      val_target_loader=None,
      device=torch.device("cpu")) -> None:

    # Initialize variables
    self.E = experiment
    self.valid_step = self.E.config.valid_step
    self.device = device

    self.batch_size = self.E.config.batch_size
    self.train_target_loader = train_target_loader
    self.val_target_loader = val_target_loader
    self.train_source_loader = train_source_loader
    self.val_source_loader = val_source_loader
    
    
    # Additional setup
    self.train_source_it = iter(train_source_loader)
    self.train_target_it = iter(train_target_loader)
    self.val_target_it = iter(val_target_loader)
    self.val_source_it = iter(val_source_loader)
    
    # Figure directory
    self.figure_dir = os.path.join(self.E.log_directory, "figures")
    os.makedirs(self.figure_dir, exist_ok=True)

    # Model
    self.model = SiameseNet2D(
      (self.E.config.channel, self.E.config.image_height, self.E.config.image_width),
      n_classes=2
    )
    print(f"n_params: {get_nparams(self.model)}")

    # Construct model path
    self.model_path = os.path.join(self.E.log_directory, "checkpoints", self.E.config.experiment_name + ".pth")
    # Path for saving training results in numpy formats
    self.history_path = os.path.join(self.E.log_directory, "training_history.npy")
    self.load_model(self.model_path)

    self.contrastive_loss = ContrastiveLoss(margin=1, device=self.device)
    
    # History profile
    self.history = Profile(self.history_path)
    self.current_epoch = self.history.get_current_epoch()

  def load_model(self, model_path):
    # Initialize model weights
    if model_path != None and os.path.exists(model_path):
      try:
        self.model.load_state_dict(torch.load(model_path))
        self.E.logger.write(f"Model weights loaded!")
      except:
        self.E.logger.write(f"Model weights not exist! Training new model...")
    self.model = self.model.to(self.device)

    self.optimizer = torch.optim.Adam(
      self.model.parameters(),
      lr=self.E.config.lr,
      weight_decay=self.E.config.weight_decay
    )

  def get_next(self, iterator, name="train_target"):
    batch = None
    try:
      batch = next(iterator)
    except:
      if name == "train_source":
        self.train_source_it = iter(self.train_source_loader)
        batch = next(self.train_source_it)
      elif name == "val_source":
        self.val_source_it = iter(self.val_source_loader)
        batch = next(self.val_source_it)
      if name == "train_target":
        self.train_target_it = iter(self.train_target_loader)
        batch = next(self.train_target_it)
      elif name == "val_target":
        self.val_target_it = iter(self.val_target_loader)
        batch = next(self.val_target_it)
    return batch

  def compute_distance(self, img0, img1):
    # input: (B, C, H, W)
    o0, o1 = self.model(img0.to(self.device), img1.to(self.device))
    return torch.nn.functional.pairwise_distance(o0, o1).cpu().numpy()

  def sample_image(self, i0, i1, o0, o1, id=0):
    
    d01 = torch.nn.functional.pairwise_distance(o0, o1)
    concat01 = torch.cat((i0[id:id+1]/2+0.5,i1[id:id+1]/2+0.5),0)
    fig = plt.figure()
    grid = torchvision.utils.make_grid(concat01)
    grid = torch.permute(grid, [1, 2, 0])
    plt.imshow(grid)
    plt.title(f"Predicted L2 Distance: {d01[id].item()}")
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(self.figure_dir, f"e_{self.current_epoch}_s_{int(step)}_same_source.png"))
    plt.close(fig)
      
  def sample_images(self, img0, img1, img2, img3, img4, img5,
      out0, out1, out2, out3, out4, out5, step):
    id = np.random.randint(0, self.batch_size)

    # This method draw and sampke images
    i0, i1, i2, i3, i4, i5, o0, o1, o2, o3, o4, o5 =\
      img0.cpu(), img1.cpu(), img2.cpu(), img3.cpu(), img4.cpu(), img5.cpu(), \
      out0.cpu(), out1.cpu(), out2.cpu(), out3.cpu(), out4.cpu(), out5.cpu()
    
    d01 = torch.nn.functional.pairwise_distance(o0, o1)
    print(f"same: {d01}")
    d23 = torch.nn.functional.pairwise_distance(o2, o3)
    print(f"same: {d23}")
    d45 = torch.nn.functional.pairwise_distance(o4, o5)
    print(f"diff: {d45}")

    concat01 = torch.cat((i0[id:id+1]/2+0.5,i1[id:id+1]/2+0.5),0)
    fig = plt.figure()
    grid = torchvision.utils.make_grid(concat01)
    grid = torch.permute(grid, [1, 2, 0])
    plt.imshow(grid)
    plt.title(f"Predicted L2 Distance: {d01[id].item()}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(self.figure_dir, f"e_{self.current_epoch}_s_{int(step)}_same_source.png"))
    plt.close(fig)

    concat23 = torch.cat((i2[id:id+1]/2+0.5,i3[id:id+1]/2+0.5),0)
    fig = plt.figure()
    grid = torchvision.utils.make_grid(concat23)
    grid = torch.permute(grid, [1, 2, 0])
    plt.imshow(grid)
    plt.title(f"Predicted L2 Distance: {d23[id].item()}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(self.figure_dir, f"e_{self.current_epoch}_s_{int(step)}_same_target.png"))
    plt.close(fig)

    concat45 = torch.cat((i4[id:id+1]/2+0.5,i5[id:id+1]/2+0.5),0)
    fig = plt.figure()
    grid = torchvision.utils.make_grid(concat45)
    grid = torch.permute(grid, [1, 2, 0])
    plt.imshow(grid)
    plt.title(f"Predicted L2 Distance: {d45[id].item()}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(self.figure_dir, f"e_{self.current_epoch}_s_{int(step)}_different.png"))
    plt.close(fig)

  @time_record
  def _train_one_epoch(self, epoch) -> List[float]:
    len_dataloader = max(len(self.train_source_loader), len(self.train_target_loader))
    self.E.logger.write(f"---------- Training epoch {epoch} ----------")
    
    for step in range(len_dataloader):
      # label indicate the dissimilarity between two image
      # The more distance, the closer to 1, the less distance, the closer to 0

      ###### TRAIN SAME PAIR IMAGE on source domain
      img0, _, _, _ = self.get_next(self.train_source_it, "train_source")
      img1, _, _, _ = self.get_next(self.train_source_it, "train_source")
      out0, out1 = self.model(img0.to(self.device), img1.to(self.device))
      # print(out0)
      label = torch.ones((self.E.config.batch_size, 1)).to(self.device)
      loss0 = self.contrastive_loss(out0, out1, label)

      ###### TRAIN SAME PAIR IMAGE on target domain
      img0, _, _, _ = self.get_next(self.train_target_it, "train_target")
      img1, _, _, _ = self.get_next(self.train_target_it, "train_target")
      out0, out1 = self.model(img0.to(self.device), img1.to(self.device))
      label = torch.ones((self.E.config.batch_size, 1)).to(self.device)
      loss1 = self.contrastive_loss(out0, out1, label)
      
      ###### TRAIN DIFFERENT PAIR IMAGE
      img0, _, _, _ = self.get_next(self.train_source_it, "train_source")
      img1, _, _, _ = self.get_next(self.train_target_it, "train_target")
      out0, out1 = self.model(img0.to(self.device), img1.to(self.device))
      label = torch.zeros((self.E.config.batch_size, 1)).to(self.device)
      loss2 = self.contrastive_loss(out0, out1, label)

      loss = 0.25 * loss0 + 0.25 * loss1 + 0.5 * loss2

      # Zero your gradients for every batch!
      self.optimizer.zero_grad()
      # Backward loss
      loss.backward()
      # Adjust learning weights
      self.optimizer.step()

      # Append loss
      self.history.add_element(int(step), PROFILE_NAME.step, epoch)
      self.history.add_element(loss.squeeze().item(), PROFILE_NAME.l, epoch)

      if step % self.valid_step == 0:
        with torch.no_grad():
          ###### TRAIN SAME PAIR IMAGE on source domain
          img0, _, _, _ = self.get_next(self.val_source_it, "val_source")
          img1, _, _, _ = self.get_next(self.val_source_it, "val_source")
          out0, out1 = self.model(img0.to(self.device), img1.to(self.device))
          label = torch.ones((self.E.config.batch_size, 1)).to(self.device)
          loss0 = self.contrastive_loss(out0, out1, label)

          self.sample_image(img0, img1, out0, out1, id=0)

          ###### TRAIN SAME PAIR IMAGE on target domain
          img2, _, _, _ = self.get_next(self.val_target_it, "val_target")
          img3, _, _, _ = self.get_next(self.val_target_it, "val_target")
          out2, out3 = self.model(img2.to(self.device), img3.to(self.device))
          label = torch.ones((self.E.config.batch_size, 1)).to(self.device)
          loss1 = self.contrastive_loss(out2, out3, label)

          self.sample_image(img2, img3, out2, out3, id=0)

          ###### TRAIN DIFFERENT PAIR IMAGE
          img4, _, _, _ = self.get_next(self.val_source_it, "val_source")
          img5, _, _, _ = self.get_next(self.val_target_it, "val_target")
          out4, out5 = self.model(img4.to(self.device), img5.to(self.device))
          label = torch.zeros((self.E.config.batch_size, 1)).to(self.device)
          loss2 = self.contrastive_loss(out4, out5, label)

          self.sample_image(img4, img5, out4, out5, id=0)

          loss = 0.25 * loss0 + 0.25 * loss1 + 0.5 * loss2
          self.history.add_element(loss.squeeze().item(), PROFILE_NAME.vl, epoch)

          self.history.log_latest_data(self.E.logger, step, total=len_dataloader, silent=False)

        self.sample_images(img0, img1, img2, img3, img4, img5,
          out0, out1, out2, out3, out4, out5, step)

  def train(self, total_epochs) -> List[np.ndarray]:

    while self.current_epoch < total_epochs:
      # Train one epoch
      self._train_one_epoch(self.current_epoch) # Train source (synthetic) dataset on feature extractors and label classifier

      # Save data
      self.history.save_numpy_data()

      # Save model
      torch.save(self.model.state_dict(), self.model_path)

      # Save a separate model every 5 epoch
      if self.current_epoch % 1 == 0:
        tmp_path = os.path.join(self.E.log_directory, "checkpoints", f"{self.E.config.experiment_name}_epoch_{self.current_epoch}.pth")
        torch.save(self.model.state_dict(), tmp_path)

      # Increment current epoch
      self.current_epoch += 1