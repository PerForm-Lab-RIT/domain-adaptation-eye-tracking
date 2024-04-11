

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow
from tqdm import tqdm
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

import h5py
from ..data import PairedImageDatasetWithLabel

from ..model.loss import StructureRetainLoss

from ..utils.experiment import PROFILE_NAME, Experiment
from ..utils.image import one_hot_label
from . import Trainer

transforms_ = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])

class CGANTrainer_v2(Trainer):
  def __init__(self, E: Experiment, G_AB, G_BA, D_A, D_B, device=torch.device("cpu"), visualization_step: int = None) -> None:
    super().__init__(E, device, visualization_step)

    # Models path
    self.ckpt_path = os.path.join(self.E.model_directory, "ckpt.pth")
    
    # model
    self.G_AB, self.G_BA, self.D_A, self.D_B =  G_AB, G_BA, D_A, D_B
    
    # optimizers
    self.optimizer_G = torch.optim.Adam(
        itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=E.config.lr_g if "lr_g" in E.config else E.config.lr, betas=(0.5,0.999)
    )
    self.optimizer_D_A = torch.optim.Adam(
        self.D_A.parameters(), lr=E.config.lr_d if "lr_d" in E.config else E.config.lr, betas=(0.5,0.999)
    )
    self.optimizer_D_B = torch.optim.Adam(
        self.D_B.parameters(), lr=E.config.lr_d if "lr_d" in E.config else E.config.lr, betas=(0.5,0.999)
    )

    # Load model and optimizers
    self.load_model()

    # Loss functions
    self.criterion_GAN = torch.nn.MSELoss().to(device)
    self.criterion_cycle = torch.nn.L1Loss().to(device)
    self.criterion_identity = torch.nn.L1Loss().to(device)
    self.criterion_edge = torch.nn.L1Loss().to(device)

    self.criterion_structure_retain = StructureRetainLoss(
      gamma_edge=E.config.gamma_edge,
      gamma_var=E.config.gamma_var,
      gamma_mean=E.config.gamma_mean,
      alpha=E.config.alpha,
      beta=E.config.beta,
      n_classes=E.config.n_classes,
      edge_threshold=E.config.edge_threshold,
      device=self.device
    )

    self.train_dataset = PairedImageDatasetWithLabel(
      E, 
      E.config.source_domain, 
      E.config.target_domain, 
      split="train", 
      transform=transforms_
    )
    self.train_loader = DataLoader(
      self.train_dataset,
      batch_size=E.config.batch_size,
      shuffle=True
    )

    self.val_dataset = PairedImageDatasetWithLabel(
      E, 
      E.config.source_domain, 
      E.config.target_domain, 
      split="validation", 
      transform=transforms_
    )
    self.val_loader = DataLoader(
      self.val_dataset,
      batch_size=E.config.batch_size,
      shuffle=True
    )
    self.val_iter = iter(self.val_loader)

    self.additional_losses = []
  
  def add_loss_object(self, loss_object: object, multiplier, name):
    self.additional_losses.append({"name": name, "loss_object": loss_object, "multiplier": multiplier})

  def save_model(self):
    # save model every epoch
    torch.save({
      "epoch": self.current_epoch,
      "G_AB": self.G_AB.state_dict(),
      "G_BA": self.G_BA.state_dict(),
      "D_A": self.D_A.state_dict(),
      "D_B": self.D_B.state_dict(),
      "Optim_G": self.optimizer_G.state_dict(),
      "Optim_DA": self.optimizer_D_A.state_dict(),
      "Optim_DB": self.optimizer_D_B.state_dict(),
    }, self.ckpt_path)

  def load_model(self, ckpt_path=None):
    _ckpt_path = None
    if ckpt_path is None:
      _ckpt_path = self.ckpt_path
    else:
      _ckpt_path = ckpt_path

    if os.path.exists(_ckpt_path):
      ckpt_data = torch.load(_ckpt_path)
      # self.current_epoch = ckpt_data["epoch"] + 1
      self.G_AB.load_state_dict(ckpt_data["G_AB"])
      self.G_BA.load_state_dict(ckpt_data["G_BA"])
      self.D_A.load_state_dict(ckpt_data["D_A"])
      self.D_B.load_state_dict(ckpt_data["D_B"])

      # https://github.com/pytorch/pytorch/issues/2830#issuecomment-514946741
      self.optimizer_G.load_state_dict(ckpt_data['Optim_G'])
      for state in self.optimizer_G.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(self.device)

      self.optimizer_D_A.load_state_dict(ckpt_data['Optim_DA'])
      for state in self.optimizer_D_A.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(self.device)

      self.optimizer_D_B.load_state_dict(ckpt_data['Optim_DB'])
      for state in self.optimizer_D_B.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(self.device)
      
    else:
      print("No weights loaded!")

    self.G_AB = self.G_AB.to(self.device)
    self.G_BA = self.G_BA.to(self.device)
    self.D_A = self.D_A.to(self.device)
    self.D_B = self.D_B.to(self.device)

  def sample_images(self, target_path=None, epoch=None, step=None):
    """show a generated sample from the test set"""
    try:
      imgs = next(self.val_iter)
    except:
      self.val_iter = iter(self.val_loader)
      imgs = next(self.val_iter)
    self.G_AB.eval()
    self.G_BA.eval()
    real_lab_A = imgs['A'][1].to(self.device)
    real_lab_B = imgs['B'][1].to(self.device)
    real_lab_hot_A = one_hot_label(real_lab_A, n_classes=self.E.config.n_classes).to(self.device)
    real_lab_hot_B = one_hot_label(real_lab_B, n_classes=self.E.config.n_classes).to(self.device)
    real_A = imgs['A'][0].to(self.device) # A : monet
    fake_B = self.G_AB(real_A, real_lab_hot_A).detach()
    real_B = imgs['B'][0].to(self.device)  # B : photo
    fake_A = self.G_BA(real_B, real_lab_hot_B).detach()

    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=4, normalize=True)
    fake_B = make_grid(fake_B, nrow=4, normalize=True)
    real_B = make_grid(real_B, nrow=4, normalize=True)
    fake_A = make_grid(fake_A, nrow=4, normalize=True)
    # Arange images along y-axis    
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    fig = plt.figure()
    plt.imshow(image_grid.cpu().permute(1,2,0))
    plt.title('Real A vs Fake B | Real B vs Fake A')
    plt.axis('off')
    if target_path is not None:
      plt.savefig(os.path.join(target_path, f"epoch_{epoch}_step_{step}"))
    else:
      plt.show()
    plt.close(fig)

  def train(self, n_epochs):

    while self.current_epoch < n_epochs:
      for i, batch in enumerate(tqdm(self.train_loader)):
          
        # Set model input
        real_A = batch['A'][0].to(self.device)
        real_B = batch['B'][0].to(self.device)

        real_lab_A = batch['A'][1].to(self.device)
        real_lab_B = batch['B'][1].to(self.device)

        real_lab_hot_A = one_hot_label(real_lab_A, n_classes=self.E.config.n_classes).to(self.device)
        real_lab_hot_B = one_hot_label(real_lab_B, n_classes=self.E.config.n_classes).to(self.device)

        # Adversarial ground truths
        valid = torch.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))).to(self.device) # requires_grad = False. Default.
        fake = torch.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))).to(self.device) # requires_grad = False. Default.
          
        # -----------------
        # Train Generators
        # -----------------
        self.G_AB.train() # train mode
        self.G_BA.train() # train mode
        
        self.optimizer_G.zero_grad() # Integrated optimizer(G_AB, G_BA)
        out1 = self.G_BA(real_A, real_lab_hot_A)
        out2 = self.G_AB(real_B, real_lab_hot_B)
        # print(out.shape)
        # print(real_A.shape)
        
        # Identity Loss
        loss_id_A = self.criterion_identity(out1, real_A) # If you put A into a generator that creates A with B,
        loss_id_B = self.criterion_identity(out2, real_B) # then of course A must come out as it is.
                                                            # Taking this into consideration, add an identity loss that simply compares 'A and A' (or 'B and B').
        loss_identity = (loss_id_A + loss_id_B)/2

        # GAN Loss
        fake_B = self.G_AB(real_A, real_lab_hot_A) # fake_B is fake-photo that generated by real monet-drawing
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid) # tricking the 'fake-B' into 'real-B'
        fake_A = self.G_BA(real_B, real_lab_hot_B)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid) # tricking the 'fake-A' into 'real-A'

        loss_GAN = (loss_GAN_AB + loss_GAN_BA)/2
        
        # Cycle Loss
        recov_A = self.G_BA(fake_B, real_lab_hot_A) # recov_A is fake-monet-drawing that generated by fake-photo
        loss_cycle_A = self.criterion_cycle(recov_A, real_A) # Reduces the difference between the restored image and the real image
        recov_B = self.G_AB(fake_A, real_lab_hot_B)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)
        
        loss_cycle = (loss_cycle_A + loss_cycle_B)/2


        # Structure retain loss
        loss_structure_retain_A = self.criterion_structure_retain(fake_B, real_A, real_lab_A, real_B, real_lab_B, recov_A)
        loss_structure_retain_B = self.criterion_structure_retain(fake_A, real_B, real_lab_B, real_A, real_lab_A, recov_B)
        loss_structure_retain = (loss_structure_retain_A + loss_structure_retain_B) / 2

        loss_G = self.E.config.gan_loss * loss_GAN + self.E.config.cycle_loss \
          * loss_cycle + self.E.config.id_loss * loss_identity + loss_structure_retain
        for loss_data in self.additional_losses:
          loss_G += loss_data["multiplier"] * loss_data["loss_object"](
            real_A=real_A, real_B=real_B, fake_A=fake_A, 
            fake_B=fake_B, real_lab_A=real_lab_A, 
            real_lab_B=real_lab_B, 
            recov_A=recov_A, recov_B=recov_B
          )

        loss_G.backward()
        self.optimizer_G.step()
          
        # -----------------
        # Train Discriminator A
        # -----------------
        self.optimizer_D_A.zero_grad()
    
        loss_real = self.criterion_GAN(self.D_A(real_A), valid) # train to discriminate real images as real
        loss_fake = self.criterion_GAN(self.D_A(fake_A.detach()), fake) # train to discriminate fake images as fake
        
        loss_D_A = (loss_real + loss_fake)/2
        
        loss_D_A.backward()
        self.optimizer_D_A.step()

        # -----------------
        # Train Discriminator B
        # -----------------
        self.optimizer_D_B.zero_grad()
    
        loss_real = self.criterion_GAN(self.D_B(real_B), valid) # train to discriminate real images as real
        loss_fake = self.criterion_GAN(self.D_B(fake_B.detach()), fake) # train to discriminate fake images as fake
        
        loss_D_B = (loss_real + loss_fake)/2
        
        loss_D_B.backward()
        self.optimizer_D_B.step()
          
        # ------> Total Loss
        loss_D = (loss_D_A + loss_D_B)/2
        
        # -----------------
        # Show Progress
        # -----------------
        if i % self.visualization_step == 0:
          self.sample_images(self.E.figure_folder, self.current_epoch, i)
          self.history.add_element(i, PROFILE_NAME.step, self.current_epoch)
          self.history.add_element(loss_D.item(), PROFILE_NAME.loss_D, self.current_epoch)
          self.history.add_element(loss_G.item(), PROFILE_NAME.loss_G, self.current_epoch)
          self.history.add_element(loss_GAN.item(), PROFILE_NAME.loss_GAN, self.current_epoch)
          self.history.add_element(loss_cycle.item(), PROFILE_NAME.loss_cycle, self.current_epoch)
          self.history.add_element(loss_identity.item(), PROFILE_NAME.loss_identity, self.current_epoch)
          self.history.add_element(loss_structure_retain.item(), PROFILE_NAME.loss_structure_retain, self.current_epoch)

          self.E.logger.write('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f - (adv : %f, cycle : %f, identity : %f, structure : %f)]'
                  %(self.current_epoch,n_epochs-1,       # [Epoch -]
                    i,len(self.train_loader),   # [Batch -]
                    loss_D.item(),       # [D loss -]
                    loss_G.item(),       # [G loss -]
                    loss_GAN.item(),     # [adv -]
                    loss_cycle.item(),   # [cycle -]
                    loss_identity.item(),# [identity -]
                    loss_structure_retain.item()
                  ))

      # save model every epoch
      torch.save({
        "epoch": self.current_epoch,
        "G_AB": self.G_AB.state_dict(),
        "G_BA": self.G_BA.state_dict(),
        "D_A": self.D_A.state_dict(),
        "D_B": self.D_B.state_dict(),
        "Optim_G": self.optimizer_G.state_dict(),
        "Optim_DA": self.optimizer_D_A.state_dict(),
        "Optim_DB": self.optimizer_D_B.state_dict(),
      }, os.path.join(self.E.model_directory, f"ckpt_epoch_{self.current_epoch}.pth"))

      self.save_model()
      self.save_history()

      self.current_epoch += 1