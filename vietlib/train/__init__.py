# %%
import os
from typing import Any
import torch
# from . import *

from abc import ABC, abstractmethod

from ..utils.experiment import Experiment, Profile

class Trainer(ABC):
  def __init__(self, E: Experiment, device=torch.device("cpu"), visualization_step: int=None) -> None:
    self.E: Experiment = E
    self.device = device
    self.visualization_step = visualization_step

    # Setup
    # Path for saving training results in numpy formats
    self.history_path = os.path.join(self.E.log_directory, "training_history.npy")
    # History profile
    self.history = Profile(self.history_path)
    self.current_epoch = self.history.get_current_epoch()

  def save_model(self):
    raise NotImplementedError

  def save_history(self):
    self.history.save_numpy_data()

def save_experiment(trainer: Trainer):
  def wrapper(func):
    func()
    # Save history
    trainer.history.save_numpy_data()
  return wrapper

class BaseDQN(ABC):
  def __init__(self) -> None:
    super().__init__()

  def step(self):
    pass

  @abstractmethod
  def _train_episode(self, episode: int):
    pass

  def train(self, episodes: int):
    # self._init_env()
    for episode in range(episodes):
      self._train_episode(episode)

def pytorch_step(loss_function, *args, optimizer: torch.optim.Optimizer=None, mode="train") -> torch.Tensor:
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
    assert optimizer is not None, "In training mode, optimizer should not be None."
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    # Compute loss and back prop
    loss = loss_function(*args)
    loss.backward()
    # Adjust learning weights
    optimizer.step()
    return loss
  elif mode == "eval":
    with torch.no_grad():
      loss = loss_function(*args)
    return loss
  else:
    raise Exception("Mode can only be train or eval")

if __name__ == "__main__":
  class DQN(BaseDQN):
    def __init__(self) -> None:
      super().__init__()
    
    def _train_episode(self, episode: int):
      super()._train_episode(episode)
      a = 3

  d = DQN()
  d.train(1)

