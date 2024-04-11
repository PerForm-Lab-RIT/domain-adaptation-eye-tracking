# Trainer code

import gc
from typing import List
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from ..model.loss import DannLossObject, SegLossObject
from ..utils.experiment import PROFILE_NAME, DannProfile, Experiment, Profile
from ..data import CVDataset
from ..utils import get_predictions, mIoU, mIoU_v2, time_record 
from ..utils.logger import Logger

class RitNetTrainer():
  def __init__(self,
      experiment_name: str, 
      model, 
      batch_size,
      train_loader, 
      val_loader,
      test_loader,
      optimizer, 
      loss_function,
      total_epochs=1,
      valid_step=10,
      log_root_folder=None,
      save_history=False,
      device=torch.device("cpu"),
      visualization_step: int=0) -> None:
    self.model = model
    self.batch_size = batch_size
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.total_epochs = total_epochs
    self.valid_step = valid_step
    self.save_history = save_history
    self.device = device
    if visualization_step != 0:
      self.visualization_step = visualization_step

    self.val_iter = iter(val_loader)
    self.test_iter = iter(test_loader)

    """ In the end, we have these path to log:
      * self.model_directory: path to model checkpoint folder: root_log_folder/experiment_name/checkpoints/
      * self.self.model_path : path to model
      * self.logger: The logger object in the file: root_log_folder/experiment_name/logs.log
      * self.history_path: root_log_folder/experiment_name/training_history.npy

    """

    # Create log directory for this experiment only
    log_directory = os.path.join(log_root_folder, experiment_name)
    os.makedirs(log_directory, exist_ok=True)

    # Path for saving trained checkpoint
    self.model_directory = os.path.join(log_directory, "checkpoints")
    os.makedirs(self.model_directory, exist_ok=True)
    self.model_path = os.path.join(log_directory, "checkpoints", experiment_name + ".pth")

    # Path for terminal outputs
    self.logger = Logger(os.path.join(log_directory, 'logs.log'))

    # Path for saving training results in numpy formats
    self.history_path = os.path.join(log_directory, "training_history.npy")

    # Path for saving figures
    self.figure_folder = os.path.join(log_directory, "figures")
    os.makedirs(self.figure_folder, exist_ok=True)

    # Initialize history
    if self.history_path != None and not os.path.exists(self.history_path):
      self.current_epoch = 0
      self.history = []
    else:
      # Read file
      self.logger.write(f"Found training history file, load data and continue training")
      with open(self.history_path, "rb") as f:
        self.history = np.load(f, allow_pickle=True).tolist()
      self.current_epoch = len(self.history)

    # Initialize model weights
    if self.model_path != None and os.path.exists(self.model_path):
      try:
        self.model.load_state_dict(torch.load(self.model_path))
        self.logger.write(f"Model weights loaded!")
      except:
        self.logger.write(f"Model weights not exist! Training new model...")

    self.alpha = np.zeros(((total_epochs)))
    self.alpha[0: np.min([125, total_epochs])] = 1 - np.arange(1, np.min([125, total_epochs]) + 1) / np.min([125, total_epochs])
    if total_epochs > 125:
      self.alpha[125:] = 1

  def _export_batch_prediction(self, batch_img, epoch, step):
    plt.figure()
    fig, axes = plt.subplots(self.batch_size, 5, figsize=(12, 40))

    with torch.no_grad():
      data = batch_img.to(self.device)
      output = self.model(data)
      predict = get_predictions(output)
      for j in range (len(batch_img)):
        pred_img_ori = predict[j].cpu().numpy()
        pred_img = pred_img_ori/3.0
        inp = batch_img[j].squeeze() * 0.5 + 0.5
        img_orig = np.clip(inp,0,1)
        img_orig = np.array(img_orig)
        
        # axes[j // 2, j % 2].figure(facecolor='white')
        axes[j,0].axis("off")
        axes[j,0].imshow(img_orig, cmap='gray')
        axes[j,0].grid(False)

        axes[j,1].axis("off")
        axes[j,1].imshow(pred_img)
        axes[j,1].grid(False)

        axes[j,2].axis("off")
        axes[j,2].imshow(pred_img_ori == 1)
        axes[j,2].grid(False)

        axes[j,3].axis("off")
        axes[j,3].imshow(pred_img_ori == 2)
        axes[j,3].grid(False)

        axes[j,4].axis("off")
        axes[j,4].imshow(pred_img_ori == 3)
        axes[j,4].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(self.figure_folder, f"result_epoch_{epoch}_step_{step}"))
    plt.close(fig)


  def _step(self, img, label, spatial_weights, distance_map, datum_id, epoch, step, mode="train") -> torch.Tensor:
    if mode == "train":

      # Prepare copying tensor to designated cpu/gpu
      img_device = img.to(self.device)
      output_device = self.model(img_device)
      label_device = label.to(self.device)

      # Zero your gradients for every batch!
      self.optimizer.zero_grad()

      # Compute loss and back prop
      loss = self.loss_function(label_device, output_device, spatial_weights, distance_map, self.alpha[epoch])
      loss.backward()
      
      # Adjust learning weights
      self.optimizer.step()
      return loss
    elif mode == "eval":
      with torch.no_grad():
        img_device = img.to(self.device)
        output_device = self.model(img_device)
        label_device = label.to(self.device)
        # Compute loss and back prop
        loss = self.loss_function(label_device, output_device, spatial_weights, distance_map, self.alpha[epoch])

      return loss, output_device
    else:
      raise Exception("Mode can only be train of eval")

  def _train_one_epoch(self, epoch) -> List[float]:

    epoch_loss_history = []
    epoch_val_loss_history = []
    mean_ious = []

    print(f"---------- Training epoch {epoch} ----------")
    for step, batch in enumerate(self.train_loader):
      # Training one step
      img, label, spatial_weights, distance_map, is_target = batch
      loss = self._step(img, label, spatial_weights, distance_map, 0, epoch, step, mode="train")
      epoch_loss_history.append(loss.item())

      # Validation and logging work per validation steps
      if step % self.valid_step == 0:

        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map, val_is_target = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map, val_is_target = next(self.val_iter)
        
        # Evaluate and Add val_loss
        val_loss, output_device = self._step(val_img, val_label, val_spatial_weights, val_distance_map, 0, epoch, step, mode="eval")
        epoch_val_loss_history.append(val_loss.item())

        # Compute mean iou and add it to the currnet epoch stats
        mean_iou = mIoU_v2(output_device.cpu(), val_label)
        mean_ious.append(mean_iou)

        # Log every valid step
        self.logger.write_silent(f"Epoch {epoch} [{step}/{len(self.train_loader)}], Loss: {loss.item()}, Val Loss: {val_loss.item()}, miou: {mean_iou}")
      
      # Display for visualization step
      if self.visualization_step and ((step % self.visualization_step) == 0):
        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map, val_is_target = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map, val_is_target = next(self.val_iter)
        self._export_batch_prediction(val_img, epoch, step)

        # TODO: GradCAM Visualization
        # TODO: Weights layer heatmap visualization

    return epoch_loss_history, epoch_val_loss_history, mean_ious

  def train(self) -> List[np.ndarray]:
    while self.current_epoch < self.total_epochs:
      # Train one epoch
      loss_history, val_loss_history, mean_ious = self._train_one_epoch(self.current_epoch)

      # Save history
      self.history.append({"loss": loss_history, "val_loss": val_loss_history, "mIoU": mean_ious})
      np.save(self.history_path, self.history)

      torch.save(self.model.state_dict(), self.model_path)

      # Increment current epoch
      self.current_epoch += 1

class RitNetTrainerMix():
  def __init__(self,
      experiment,
      experiment_name: str, 
      model, 
      batch_size,
      train_loader, 
      train_loader_2,
      val_dataset_target,
      val_loader,
      val_loader_2,
      test_loader,
      optimizer, 
      scheduler,
      loss_function,
      total_epochs=1,
      valid_step=10,
      log_root_folder=None,
      save_history=False,
      device=torch.device("cpu"),
      visualization_step: int=0) -> None:
    self.model = model
    self.batch_size = batch_size
    self.train_loader = train_loader # first loader is target domain because we validate on this
    self.train_loader_2 = train_loader_2 # second is source domain
    self.val_loader = val_loader
    self.val_loader_2 = val_loader_2
    self.test_loader = test_loader
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.total_epochs = total_epochs
    self.valid_step = valid_step
    self.save_history = save_history
    self.val_dataset_target = val_dataset_target
    self.device = device
    self.scheduler = scheduler
    if visualization_step != 0:
      self.visualization_step = visualization_step
    self.train_iter = iter(self.train_loader)
    self.train_iter_2 = iter(self.train_loader_2)
    self.val_iter = iter(val_loader)
    self.val_iter_2 = iter(val_loader_2)
    self.test_iter = iter(test_loader)

    self.E = experiment
    # Create log directory for this experiment only
    log_directory = self.E.log_directory

    # Path for saving trained checkpoint
    self.model_directory = os.path.join(log_directory, "checkpoints")
    os.makedirs(self.model_directory, exist_ok=True)
    self.model_path = os.path.join(log_directory, "checkpoints", experiment_name + ".pth")

    # Path for terminal outputs
    self.logger = Logger(os.path.join(log_directory, 'logs.log'))

    # Path for saving training results in numpy formats
    self.history_path = os.path.join(log_directory, "training_history.npy")

    # Path for saving figures
    self.figure_folder = os.path.join(log_directory, "figures")
    os.makedirs(self.figure_folder, exist_ok=True)

    # Prev val loss setup. # NOTE: This is not saved, so this statistic does not support multiple time training
    self.prev_val_loss = 1e8

    # Initialize history
    if self.history_path != None and not os.path.exists(self.history_path):
      self.current_epoch = 0
      self.history = []
    else:
      # Read file
      self.logger.write(f"Found training history file, load data and continue training")
      with open(self.history_path, "rb") as f:
        self.history = np.load(f, allow_pickle=True).tolist()
      self.current_epoch = len(self.history)

    # Initialize model weights
    if self.model_path != None and os.path.exists(self.model_path):
      try:
        self.model.load_state_dict(torch.load(self.model_path))
        self.logger.write(f"Model weights loaded!")
      except:
        self.logger.write(f"Model weights not exist! Training new model...")

    self.alpha = np.zeros(((total_epochs)))
    self.alpha[0: np.min([125, total_epochs])] = 1 - np.arange(1, np.min([125, total_epochs]) + 1) / np.min([125, total_epochs])
    if total_epochs > 125:
      self.alpha[125:] = 1

  def _export_batch_prediction(self, batch_img, epoch, step):
    plt.figure()
    fig, axes = plt.subplots(self.batch_size, 5, figsize=(12, 40))

    with torch.no_grad():
      data = batch_img.to(self.device)
      output = self.model(data)
      predict = get_predictions(output)
      for j in range (len(batch_img)):
        pred_img_ori = predict[j].cpu().numpy()
        pred_img = pred_img_ori/3.0
        inp = batch_img[j].squeeze() * 0.5 + 0.5
        img_orig = np.clip(inp,0,1)
        img_orig = np.array(img_orig)
        
        # axes[j // 2, j % 2].figure(facecolor='white')
        axes[j,0].axis("off")
        axes[j,0].imshow(img_orig, cmap='gray')
        axes[j,0].grid(False)

        axes[j,1].axis("off")
        axes[j,1].imshow(pred_img)
        axes[j,1].grid(False)

        axes[j,2].axis("off")
        axes[j,2].imshow(pred_img_ori == 1)
        axes[j,2].grid(False)

        axes[j,3].axis("off")
        axes[j,3].imshow(pred_img_ori == 2)
        axes[j,3].grid(False)

        axes[j,4].axis("off")
        axes[j,4].imshow(pred_img_ori == 3)
        axes[j,4].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(self.figure_folder, f"result_epoch_{epoch}_step_{step}"))
    plt.close(fig)

  def _step(self, img, label, spatial_weights, distance_map, datum_id, epoch, step, mode="train") -> torch.Tensor:
    if mode == "train":

      # Prepare copying tensor to designated cpu/gpu
      img_device = img.to(self.device)
      output_device = self.model(img_device)
      label_device = label.to(self.device)

      # Zero your gradients for every batch!
      self.optimizer.zero_grad()

      # Compute loss and back prop
      loss = self.loss_function(label_device, output_device, spatial_weights, distance_map, self.alpha[epoch])
      loss.backward()
      
      # Adjust learning weights
      self.optimizer.step()
      return loss
    elif mode == "eval":
      with torch.no_grad():
        img_device = img.to(self.device)
        output_device = self.model(img_device)
        label_device = label.to(self.device)
        # Compute loss and back prop
        loss = self.loss_function(label_device, output_device, spatial_weights, distance_map, self.alpha[epoch])

      return loss, output_device
    else:
      raise Exception("Mode can only be train of eval")

  def _train_one_epoch(self, epoch) -> List[float]:

    epoch_loss_history = []
    epoch_val_loss_history = []
    mean_ious = []
    self.train_iter = iter(self.train_loader)
    self.train_iter_2 = iter(self.train_loader_2)
    print(f"---------- Training epoch {epoch} ----------")
    step = 0
    out_ds_1 = False
    out_ds_2 = False
    while True:
      if out_ds_1 and out_ds_2:
        break
      try:
        # Training one step
        img, label, spatial_weights, distance_map = next(self.train_iter)
        loss = self._step(img, label, spatial_weights, distance_map, 0, epoch, step, mode="train")
        epoch_loss_history.append(loss.item())
        step += 1
      except:
        out_ds_1 = True
      
      try:
        # Training one step
        img, label, spatial_weights, distance_map = next(self.train_iter_2)
        loss = self._step(img, label, spatial_weights, distance_map, 0, epoch, step, mode="train")
        epoch_loss_history.append(loss.item())
        step += 1
      except:
        out_ds_2 = True

      # Validation and logging work per validation steps
      if step % self.valid_step == 0:

        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        
        # Evaluate and Add val_loss
        val_loss, output_device = self._step(val_img, val_label, val_spatial_weights, val_distance_map, 0, epoch, step, mode="eval")
        epoch_val_loss_history.append(val_loss.item())

        # Compute mean iou and add it to the currnet epoch stats
        mean_iou = mIoU_v2(output_device.cpu(), val_label)
        mean_ious.append(mean_iou)

        # Log every valid step
        self.logger.write_silent(f"Epoch {epoch} [{step}/{len(self.train_loader)+len(self.train_loader_2)}], Loss: {loss.item()}, Val Loss: {val_loss.item()}, miou: {mean_iou}")
      
      # Display for visualization step
      # Currnetly it's too many so we only export once at the start of the epoch, 
      # if self.visualization_step and ((step % self.visualization_step) == 0):
      if step == 0 and epoch % 10 == 0:
        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        self._export_batch_prediction(val_img, epoch, step)

        # TODO: GradCAM Visualization
        # TODO: Weights layer heatmap visualization

      # step += 1

    return epoch_loss_history, epoch_val_loss_history, mean_ious

  def validate_dataset(self, epoch):

    losses = []

    # for i in range(len(self.val_dataset_target)):
    for i, batch in enumerate(self.val_loader):
      if i % 100 == 0:
        print(f"Validate batch {i}")
      val_img, val_label, val_spatial_weights, val_distance_map = batch

      # Evaluate and Add val_loss
      val_loss, _ = self._step(val_img, val_label, val_spatial_weights, val_distance_map, 0, epoch, 1, mode="eval")
      losses.append(val_loss.item()) # HKS: add the mean loss of the batch instead of a single loss to the array

    return np.mean(losses)

  def train(self) -> List[np.ndarray]:
    while self.current_epoch < self.total_epochs:
      # Train one epoch
      loss_history, val_loss_history, mean_ious = self._train_one_epoch(self.current_epoch)

      # VALIDATE THE WHOLE THING
      if self.scheduler is not None:
        val_loss = self.validate_dataset(self.current_epoch)
        self.scheduler.step(val_loss)
        self.E.logger.write_silent(f"current lr after scheduler step: {self.scheduler.optimizer.param_groups[0]['lr']}")
        
        if val_loss < self.prev_val_loss:
          self.E.logger.write_silent(f"Best model at epoch {self.current_epoch}")
          torch.save(self.model.state_dict(), os.path.join(self.E.log_directory, "checkpoints", f"best_model.pth"))
          self.prev_val_loss = val_loss

      # Save history
      self.history.append({"loss": loss_history, "val_loss": val_loss_history, "mIoU": mean_ious})
      np.save(self.history_path, self.history)
      torch.save(self.model.state_dict(), self.model_path)
      if self.current_epoch % 50 == 0:
          tmp_path = os.path.join(self.E.log_directory, "checkpoints", f"{self.E.config.experiment_name}_epoch_{self.current_epoch}.pth")
          torch.save(self.model.state_dict(), tmp_path)

      # Increment current epoch
      self.current_epoch += 1

class DannTrainer():

  def __init__(self,
      experiment: Experiment,
      model,
      train_loader=None, 
      val_loader=None,
      test_loader=None,
      optimizer=None,
      device=torch.device("cpu"),
      visualization_step: int=0) -> None:
    self.E = experiment

    self.model = model

    self.batch_size = self.E.config.batch_size

    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader

    self.optimizer = optimizer
    self.loss_function = DannLossObject(device) # For loss function in the class classifer
    self.valid_step = self.E.config.valid_step
    self.device = device
    if visualization_step != 0:
      self.visualization_step = visualization_step

    self.val_iter = iter(val_loader)

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
    
    # History profile
    self.history = DannProfile(self.history_path)
    self.current_epoch = self.history.get_current_epoch()

  def _export_batch_prediction(self, batch_img, is_target_domain, epoch, step):
    fig = plt.figure(figsize=(20, 35))
    # fig, axes = plt.subplots(self.batch_size, ncols=1, constrained_layout=True, figsize=(12, 40))

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
    plt.savefig(os.path.join(self.E.figure_folder, f"result_epoch_{epoch}_step_{step}"))
    plt.close(fig)

  def _step(self, img, label, spatial_weights, distance_map, datum_id, is_target_domain, epoch, step, mode="train") -> torch.Tensor:
    """ flattened is the tensor after the reverse gradient layer

    Args:
        img (_type_): _description_
        label (_type_): _description_
        spatial_weights (_type_): _description_
        distance_map (_type_): _description_
        datum_id (_type_): _description_
        is_target_domain (bool): _description_
        epoch (_type_): _description_
        step (_type_): _description_
        mode (str, optional): _description_. Defaults to "train".

    Raises:
        Exception: _description_

    Returns:
        torch.Tensor: _description_
    """
    if mode == "train":
      # Prepare copying tensor to designated cpu/gpu
      img_device = img.to(self.device)
      output_device, domain_out_device, latent, el_params, flattened = self.model(img_device, self.alpha[epoch], self.backward_alpha[epoch])
      label_device = label.to(self.device)
      is_target_domain_device = is_target_domain.to(self.device)
      
      # Zero your gradients for every batch!
      self.optimizer.zero_grad()

      # Compute loss and back prop
      loss, seg_loss, domain_loss = self.loss_function(label_device, output_device, spatial_weights, distance_map, is_target_domain_device, domain_out_device, self.alpha[epoch])
      loss.backward()
      

      saved_l2_after_reg_grad = None
      bf = None
      try:
        # Extract gradient
        saved_l2_after_reg_grad: torch.Tensor = torch.clone(self.model.domain_classifier.lin1.weight.grad).cpu().detach()
        bf: torch.Tensor = torch.clone(self.model.down_block5.bn.weight.grad).cpu().detach()
      except:
        pass

      # Adjust learning weights
      self.optimizer.step()

      return loss, seg_loss, domain_loss, saved_l2_after_reg_grad, bf  #, ub110wg

    elif mode == "eval":
      with torch.no_grad():
        img_device = img.to(self.device)
        output_device, domain_out_device, latent, el_params, flattened = self.model(img_device, self.alpha[epoch], self.backward_alpha[epoch])
        label_device = label.to(self.device)
        is_target_domain_device = is_target_domain.to(self.device)
        # Compute loss and back prop
        loss, seg_loss, domain_loss = self.loss_function(label_device, output_device, spatial_weights, distance_map, is_target_domain_device, domain_out_device, self.alpha[epoch])

      return loss, seg_loss, domain_loss, output_device, domain_out_device

    else:
      raise Exception("Mode can only be train or eval")

  def _train_one_epoch(self, epoch) -> List[float]:

    print(f"---------- Training epoch {epoch} ----------")
    for step, batch in enumerate(self.train_loader):
      # Training one step
      img, label, spatial_weights, distance_map, datum_id, is_target_domain = batch
      loss, seg_loss, domain_loss, flattened_grad, bf = self._step(img, label, spatial_weights, distance_map, datum_id, is_target_domain, epoch, step, mode="train")
      
      # Append loss
      self.history.add_element(loss.squeeze().item(), PROFILE_NAME.l, epoch)
      self.history.add_element(seg_loss.squeeze().item(), PROFILE_NAME.sl, epoch)
      self.history.add_element(domain_loss.squeeze().item(), PROFILE_NAME.dl, epoch)

      try:
        # compute l2 gradient for the output after the reverse gradient layer
        l2_grad_after_rg: float = np.linalg.norm(flattened_grad.numpy(), ord="fro", axis=None) # shape (B, features) -> float
        self.history.add_element(l2_grad_after_rg, PROFILE_NAME.l2gar, epoch)
      except:
        pass

      try:
        l2bf: float = np.linalg.norm(bf.numpy()) # shape (B, features) -> float
        self.history.add_element(l2bf, PROFILE_NAME.l2bf, epoch)
      except:
        pass

      # Validation and logging work per validation steps
      if step % self.valid_step == 0:

        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map, val_datum_id, val_is_target_domain = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map, val_datum_id, val_is_target_domain = next(self.val_iter)
        
        # Evaluate and Add val_loss
        val_loss, seg_val_loss, domain_val_loss, output_device, domain_out_device = self._step(val_img, val_label, val_spatial_weights, val_distance_map, val_datum_id, val_is_target_domain, epoch, step, mode="eval")
        self.history.add_element(val_loss.squeeze().item(), PROFILE_NAME.vl, epoch)
        self.history.add_element(seg_val_loss.squeeze().item(), PROFILE_NAME.vsl, epoch)
        self.history.add_element(domain_val_loss.squeeze().item(), PROFILE_NAME.vdl, epoch)

        # Compute mean iou and add it to the currnet epoch stats
        mean_iou: float = mIoU_v2(output_device.cpu(), val_label)
        self.history.add_element(mean_iou, PROFILE_NAME.miou, epoch)

        # Log every valid step
        self.history.log_latest_data(self.E.logger, step, len(self.train_loader), silent=True)

      # print(self.visualization_step)
      # Display for visualization step
      if self.visualization_step and ((step % self.visualization_step) == 0):
        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map, val_datum_id, val_is_target_domain = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map, val_datum_id, val_is_target_domain = next(self.val_iter)
        self._export_batch_prediction(val_img, val_is_target_domain, epoch, step)

        # TODO: GradCAM Visualization
        # TODO: Weights layer heatmap visualization

  def train(self, total_epochs) -> List[np.ndarray]:
    # Setup alpha
    self.alpha = np.zeros(((total_epochs)))
    self.alpha[0: np.min([125, total_epochs])] = 1 - np.arange(1, np.min([125, total_epochs]) + 1) / np.min([125, total_epochs])
    if total_epochs > 125:
      self.alpha[125:] = 1

    # Setup backward alpha
    # For the first 50 epoch, does not include the graident of the discriminator that much.
    self.backward_alpha = np.zeros(((total_epochs)))
    self.backward_alpha[0: np.min([10, total_epochs])] = 1 - np.arange(1, np.min([10, total_epochs]) + 1) / np.min([10, total_epochs])
    if total_epochs > 10:
      self.backward_alpha[10:] = 1

    while self.current_epoch < total_epochs:
      # Train one epoch
      self._train_one_epoch(self.current_epoch)
      self.history.save_numpy_data()
      torch.save(self.model.state_dict(), self.model_path)

      # Increment current epoch
      self.current_epoch += 1

class RITnetTrainer_v2():
  """ TODO: Gradcam: https://github.com/jacobgil/pytorch-grad-cam
  """
  def __init__(self,
      experiment: Experiment,
      model,
      train_loader=None,
      val_dataset=None,
      val_loader=None,
      optimizer=None,
      scheduler=None,
      device=torch.device("cpu"),
      visualization_step: int=None) -> None:

    # Initialize variables
    self.E = experiment
    self.model = model
    self.batch_size = self.E.config.batch_size
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.valid_step = self.E.config.valid_step
    self.device = device
    self.visualization_step = visualization_step
    self.val_dataset = val_dataset
    self.scheduler = scheduler
    
    # Prev val loss setup. # NOTE: This is not saved, so this statistic does not support multiple time training
    self.prev_val_loss = 1e8

    # Additional setup
    self.val_iter = iter(val_loader)
    self.loss_obj = SegLossObject(device)

    # Construct model path
    self.model_path = os.path.join(self.E.model_directory, self.E.config.experiment_name + ".pth")
    # Path for saving training results in numpy formats
    self.history_path = os.path.join(self.E.log_directory, "training_history.npy")
    # Initialize model weights
    if self.model_path != None and os.path.exists(self.model_path):
      try:
        self.model.load_state_dict(torch.load(self.model_path))
        self.E.logger.write(f"Model weights loaded!")
      except:
        self.E.logger.write(f"Model weights not exist! Training new model...")
    # History profile
    self.history = Profile(self.history_path)
    self.current_epoch = self.history.get_current_epoch()

  def _export_batch_prediction(self, batch_img, epoch, step, is_target_domain=None, name="seg"):
    fig = plt.figure(figsize=(20, 35))
    # fig, axes = plt.subplots(self.batch_size, ncols=1, constrained_layout=True, figsize=(12, 40))

    subfigs = fig.subfigures(nrows=self.batch_size, ncols=1)

    with torch.no_grad():
      data = batch_img.to(self.device)
      output = self.model(data)
      predict = get_predictions(output)

      for j, subfig in enumerate(subfigs):

        pred_img_ori = predict[j].cpu().numpy()
        pred_img = pred_img_ori/3.0
        inp = batch_img[j].squeeze() * 0.5 + 0.5
        img_orig = np.clip(inp,0,1)
        img_orig = np.array(img_orig)

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

    # https://stackoverflow.com/a/1316793
    # gc.collect()

  def _step(self, loss_function, *args, mode="train") -> torch.Tensor:
    """ Perform one optimization step

    Args:
        loss_function (_type_): _description_
        mode (str, optional): _description_. Defaults to "train".

    Raises:
        Exception: _description_

    Returns:
        torch.Tensor: _description_
    """
    if mode == "train":
      
      # Zero your gradients for every batch!
      self.optimizer.zero_grad()

      # Compute loss and back prop
      loss = loss_function(*args)
      loss.backward()

      # Adjust learning weights
      self.optimizer.step()

      return loss

    elif mode == "eval":
      with torch.no_grad():
        loss = loss_function(*args)
      return loss

    else:
      raise Exception("Mode can only be train or eval")

  @time_record
  def _train_one_epoch(self, epoch) -> List[float]:

    self.E.logger.write_silent(f"---------- Training epoch {epoch} on source data (label classfier) ----------")
    for step, batch in enumerate(self.train_loader):
      # Training one step. Batch extract
      img, label, spatial_weights, distance_map = batch

      # Prepare copying tensor to designated cpu/gpu. Model inference
      img_device = img.to(self.device)
      output_device = self.model(img_device)
      label_device = label.to(self.device)

      # Compute loss
      loss = self._step(self.loss_obj, label_device, output_device, spatial_weights, distance_map, self.alpha[epoch], mode="train")
  
      # Append loss
      self.history.add_element(loss.squeeze().item(), PROFILE_NAME.sl, epoch)

      # Append gradient for seg
      bf: torch.Tensor = torch.clone(self.model.down_block5.bn.weight.grad).cpu().detach()
      l2bf: float = np.linalg.norm(bf.numpy()) # shape (B, features) -> float
      self.history.add_element(l2bf, PROFILE_NAME.l2bf_s, epoch)

      del label_device, output_device, img_device, img, label, \
        spatial_weights, distance_map, bf, l2bf
      # https://stackoverflow.com/a/1316793
      # gc.collect()

      # Validation and logging work per validation steps
      if step % self.valid_step == 0:

        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        
        with torch.no_grad():
          # Prepare copying tensor to designated cpu/gpu. Model inference
          val_img_device = val_img.to(self.device)
          val_output_device = self.model(val_img_device)
          val_label_device = val_label.to(self.device)
        
        # Evaluate and Add val_loss
        val_loss = self._step(self.loss_obj, val_label_device, val_output_device, val_spatial_weights, val_distance_map, self.alpha[epoch], mode="eval")
        self.history.add_element(val_loss.squeeze().item(), PROFILE_NAME.vsl, epoch)

        # Compute mean iou and add it to the currnet epoch stats
        mean_iou: float = mIoU_v2(val_output_device.cpu(), val_label)
        self.history.add_element(mean_iou, PROFILE_NAME.miou_s, epoch)

        # Log every valid step
        self.history.log_latest_data(self.E.logger, step, len(self.train_loader), silent=True)

        del val_output_device, val_label_device, val_loss, \
          mean_iou, val_img_device, val_img, val_label, val_spatial_weights, \
          val_distance_map
        # https://stackoverflow.com/a/1316793
        # gc.collect()

      if step == 0 and epoch % 10 == 0:
        # StopIteration issue when using iterator to get only next batch
        # https://github.com/pytorch/pytorch/issues/1917
        try:
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        except:
          self.val_iter = iter(self.val_loader)
          val_img, val_label, val_spatial_weights, val_distance_map = next(self.val_iter)
        self._export_batch_prediction(val_img, epoch, step, name="seg")

        # Memory optimization
        # free_memories(["val_img", "val_label", "val_spatial_weights", "val_distance_map"])
        del val_img, val_label, val_spatial_weights, val_distance_map
        # https://stackoverflow.com/a/1316793
        # gc.collect()

  def validate_dataset(self, epoch):

    losses = []

    for i in range(len(self.val_dataset)):

      if i % 100 == 0:
        print(f"datapoint {i}")

      val_img, val_label, val_spatial_weights, val_distance_map = self.val_dataset[i]
      # Prepare copying tensor to designated cpu/gpu. Model inference
      val_img_device = val_img.to(self.device).unsqueeze(0)
      val_output_device = self.model(val_img_device)
      val_label_device = val_label.to(self.device).unsqueeze(0)
      val_spatial_weights = val_spatial_weights.unsqueeze(0)
      val_distance_map = val_distance_map.unsqueeze(0)

      # Evaluate and Add val_loss
      val_loss = self._step(self.loss_obj, val_label_device, val_output_device, val_spatial_weights, val_distance_map, self.alpha[epoch], mode="eval")
      losses.append(val_loss.item())

    min_l = np.min(losses)
    # print(f"min: {min_l}")
    max_l = np.max(losses)
    # print(f"max: {max_l}")
    mean_l = np.mean(losses)
    # print(f"mean: {mean_l}")
    return mean_l

  def train(self, total_epochs, save_model=True, save_history=True) -> List[np.ndarray]:
    # Setup alpha
    self.alpha = np.zeros(((total_epochs)))
    self.alpha[0: np.min([125, total_epochs])] = 1 - np.arange(1, np.min([125, total_epochs]) + 1) / np.min([125, total_epochs])
    if total_epochs > 125:
      self.alpha[125:] = 1

    while self.current_epoch < total_epochs:
      # Train one epoch
      self._train_one_epoch(self.current_epoch) # Train source (synthetic) dataset on feature extractors and label classifier

      # VALIDATE THE WHOLE THING
      if self.scheduler is not None:
        val_loss = self.validate_dataset(self.current_epoch)
        self.scheduler.step(val_loss)
        self.E.logger.write(f"val_loss: {val_loss}, current lr after scheduler step: {self.scheduler.optimizer.param_groups[0]['lr']}")
        
        if val_loss < self.prev_val_loss:
          self.E.logger.write(f"Best model at epoch {self.current_epoch}")
          torch.save(self.model.state_dict(), os.path.join(self.E.log_directory, "checkpoints", f"best_model.pth"))
          self.prev_val_loss = val_loss

      if save_history:
        # Save data
        self.history.save_numpy_data()

      if save_model:
        torch.save(self.model.state_dict(), self.model_path)
        # Save a separate model every 5 epoch
        if self.current_epoch % 50 == 0:
          tmp_path = os.path.join(self.E.log_directory, "checkpoints", f"{self.E.config.experiment_name}_epoch_{self.current_epoch}.pth")
          torch.save(self.model.state_dict(), tmp_path)

      # Increment current epoch
      self.current_epoch += 1

class RITnetTrainer_v2_custom_saved_path(RITnetTrainer_v2):
  def __init__(self, 
      experiment: Experiment,
      history_path,
      model_path,
      model, 
      train_loader=None, 
      val_loader=None, 
      optimizer=None, 
      device=None, 
      visualization_step: int = None
    ) -> None:
    # Initialize variables
    self.E = experiment
    self.model = model
    self.batch_size = self.E.config.batch_size
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.valid_step = self.E.config.valid_step
    self.device = device
    self.visualization_step = visualization_step

    self.model_path = model_path
    self.history_path = history_path  

    # Additional setup
    self.val_iter = iter(val_loader)
    self.loss_obj = SegLossObject(device)

    if self.model_path != None and os.path.exists(self.model_path):
      try:
        self.model.load_state_dict(torch.load(self.model_path))
        self.E.logger.write(f"Model weights loaded!")
      except:
        self.E.logger.write(f"Model weights not exist! Training new model...")
    # History profile
    self.history = Profile(self.history_path)
    self.current_epoch = self.history.get_current_epoch()

