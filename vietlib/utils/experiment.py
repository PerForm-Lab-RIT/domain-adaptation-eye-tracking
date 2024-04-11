# %%

import collections
import json
from typing import List
# from ritnet.logger import Logger
import os
from munch import Munch
import numpy as np
from .logger import Logger
import warnings
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import json
from argparse import Namespace
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

# These name have to be different because these names are used as key in other schemes.
PROFILE_NAME: Munch = Munch.fromDict({
  "step": "step",
  "l": "Loss",
  "al": "Actor Loss",
  "cl": "Critic Loss",
  "tl": "Termination Loss",
  "vl": "Val Loss",
  "sl": "Segment Loss",
  "dl": "Domain Loss",
  "ssl": "Source Domain's Segment Loss",
  "sdl": "Source Domain's Domain Loss",
  "tdl": "Target Domain's Domain Loss",
  "vsl": "Val Segment Loss",
  "vdl": "Val Domain Loss",
  "l2dg": "L2 Domain Grad",
  "miou": "Mean IoU",
  "l2gar": "L2 of After Reverse Gradient's Grad",
  "l2seg": "L2 of First Up Block",
  "l2bf": "L2 of Before Reverse Gradient's Grad",
  "miou_s": "Mean IoU when training Seg Net",
  "miou_d": "Mean IoU when training Dom Net",
  "miou_on_target": "Mean IoU on target domain",
  "v_miou_on_target": "Val mean IoU on target domain",
  "l2bf_d": "L2 of Before Reverse Gradient's Grad Dom",
  "l2bf_s": "L2 of Before Reverse Gradient's Grad Seg",
  "rw": "Reward",
  "episode": "episode",
  "miou_rw": "mIoU of the reward set",
  "mmiou_rw": "Environment mIoU of the reward set",
  "loss_D": "discriminator loss", #loss_D.item(),       # [D loss -]
  "loss_G": "generator loss", # loss_G.item(),       # [G loss -]
  "loss_GAN": "adversarial", # loss_GAN.item(),     # [adv -]
  "loss_cycle": "cycle", # loss_cycle.item(),   # [cycle -]
  "loss_identity": "identity", # loss_identity.item(),# [identity -]
  "loss_id_edge": "edge", # loss_id_edge.item(),
  "loss_geo": "geomask", # loss_geo.item(),
  "loss_structure_retain": "structure"
})

class Experiment(ABC):
  def __init__(self, json_path: str, fold=None, log_root_folder=None) -> None:
    self.config: Munch = self._load_config_from_json(json_path)
    
    if log_root_folder is None:
      self.log_root_folder = self.config.log_root_folder
    else:
      self.log_root_folder = log_root_folder

    # Create log root folder if it does not exist!
    os.makedirs(self.log_root_folder, exist_ok=True)

    # Create log directory for this experiment only
    if fold is None:
      self.log_directory = os.path.join(self.log_root_folder, self.config.experiment_name)
      os.makedirs(self.log_directory, exist_ok=True)
    else:
      self.log_directory = os.path.join(self.log_root_folder, self.config.experiment_name, f"fold_{fold}")
      os.makedirs(self.log_directory, exist_ok=True)

    # Path for terminal outputs
    self.logger = Logger(os.path.join(self.log_directory, 'logs.log'))

  def _load_config_from_json(self, json_path):
    # parse the configurations from the config json file provided
    with open(json_path, 'r') as config_file:
      config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = Munch.fromDict(config_dict)
    return config

  @abstractmethod
  def run_experiment(self):
    pass

  @abstractmethod
  def save_experiment(self):
    pass


class Experiment_v2():
  """ New version of experiment, this will not take the config any more, instead it takes only the 
  log root folder
  """
  def __init__(self, log_root_folder, experiment_name, fold=None) -> None:
    """ Initialize

    Args:
      log_root_folder (_type_): _description_
      experiment_name (_type_): _description_
      fold (_type_, optional): _description_. Defaults to None.
    """
    self.log_root_folder = log_root_folder
    # Create log root folder if it does not exist!
    os.makedirs(self.log_root_folder, exist_ok=True)

    # Create log directory for this experiment only
    if fold is None:
      self.log_directory = os.path.join(self.log_root_folder, experiment_name)
    else:
      self.log_directory = os.path.join(self.log_root_folder, experiment_name, f"fold_{fold}")
    os.makedirs(self.log_directory, exist_ok=True)

    # Path for terminal outputs
    self.logger = Logger(os.path.join(self.log_directory, 'logs.log'))

    # Path for tensorboard pytorch logging
    # self.tensorboard_logger: SummaryWriter = SummaryWriter(log_dir=os.path.join(self.log_directory, "tensorboard"))
    assert experiment_name != "tboard"
    if fold is None:
      tboard_logdir = os.path.join(log_root_folder, "tboard", experiment_name)
    else:
      tboard_logdir = os.path.join(log_root_folder, "tboard", experiment_name, f"fold_{fold}")
    self.tboard = tf.summary.create_file_writer(tboard_logdir)

    # Path for saving trained checkpoint
    self.model_directory = os.path.join(self.log_directory, "checkpoints")
    os.makedirs(self.model_directory, exist_ok=True)
    self.checkpoint_path_tf = os.path.join(self.model_directory, "ckpt")
    self.checkpoint_path_pth = os.path.join(self.model_directory, "ckpt.pth")

    # Path for saving figures
    self.figure_folder = os.path.join(self.log_directory, "figures")
    os.makedirs(self.figure_folder, exist_ok=True)

    # Path for additional profile logging
    self.history_path = os.path.join(self.log_directory, "training_history.npy")
    self.history = Profile(self.history_path)

    # Config variable for deprecated support, this will be initialized when the
    # argument is passed in the `on_experiment_start(self, args: Namespace)` function
    self.config = None

  def on_experiment_start(self, args: Namespace):
    variable_dictionary = vars(args)
    self.config = Munch(variable_dictionary)
    with open(os.path.join(self.log_directory, "config.json"), 'wt') as f:
      json.dump(variable_dictionary, f, indent=2)

  def on_experiment_end(self):
    # self.tensorboard_logger.close()
    pass

  def log_data(self, stats: float, key: str, episode=None, step=None):
    if episode is not None:
      self.history.add_element(stats, key, episode)

    # if step is not None:
    #   self.tensorboard_logger.add_scalar(key, stats, step)

class MLExperiment(Experiment):
  def __init__(self, json_path: str, fold=None, log_root_folder=None) -> None:
    super().__init__(json_path, fold=fold, log_root_folder=log_root_folder)

    # Path for saving trained checkpointz
    self.model_directory = os.path.join(self.log_directory, "checkpoints")
    os.makedirs(self.model_directory, exist_ok=True)

    # Path for terminal outputs
    self.logger = Logger(os.path.join(self.log_directory, 'logs.log'))

    # Path for saving figures
    self.figure_folder = os.path.join(self.log_directory, "figures")
    os.makedirs(self.figure_folder, exist_ok=True)
  
  def run_experiment(self):
    pass

  def save_experiment(self):
    pass

class CVExperiment(MLExperiment):
  def __init__(self, json_path: str, fold=None, log_root_folder=None) -> None:
    super().__init__(json_path, fold=fold, log_root_folder=log_root_folder)

    # Setup other global variables
    self.config.training_image_size = (self.config.image_size.height, self.config.image_size.width)
    self.h_over_w_ratio = self.config.image_size.height / self.config.image_size.width

  def run_experiment(self):
    pass

  def save_experiment(self):
    pass

class Profile:
  def __init__(self, history_path: str=None) -> None:
    """ Expect a numpy path

    Args:
        history_path (str, optional): _description_. Defaults to None.
    """
    # [{}, {}, {}] history of epoch data
    self.history = []
    self.history_path: str = history_path
    if self.history_path != None and os.path.exists(self.history_path):
      with open(self.history_path, "rb") as f:
        self.history = np.load(f, allow_pickle=True).tolist()
    
  def plot_data(self, data_key: str, xlabel="", ylabel="", window_size=0) -> None:

    data = []

    for epoch, epoch_data in enumerate(self.history):
      # if data_key in epoch_data:
      data = data + epoch_data[data_key]
    # x = np.linspace(1, len(data) + 1, len(data))
    if len(data) > 0:
      if window_size == 0:
        fig = plt.figure()
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
      else:
        import pandas as pd
        df = pd.DataFrame(data)
        df['avg'] = df.iloc[:, 0].rolling(window_size, min_periods=1).mean()
        plt.plot(np.linspace(1,df['avg'].shape[0]+1,df['avg'].shape[0]), df['avg'])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    else:
      print("No data to plot")

  def get_current_epoch(self):
    return len(self.history)

  def get_current_global_step(self, indicator: str):
    """

    Args:
      indicator (str): PROFILE_NAME's key to indicate which key in the
        history disctionary to consider as step  
    """
    n_steps = 0
    for epoch_data in self.history:
      n_steps += len(epoch_data[indicator])
    return n_steps
    
  def add_element(self, data, name: str, epoch: int) -> None:
    """ Add element dynamically

    Args:
      data (_type_): _description_
      name (str): _description_
      epoch (int): _description_

    Raises:
      Exception: _description_
    """
    if data is None:
      return
      
    if epoch == len(self.history):
      self.history.append(collections.defaultdict(list))
      self.history[epoch][name].append(data)
    elif epoch >= 0 and epoch < len(self.history):
      self.history[epoch][name].append(data)
    else:
      raise Exception("Error adding data to loss.")

  def save_numpy_data(self) -> None:
    np.save(self.history_path, self.history)

  def log_latest_data(self, logger: Logger, step: int, total: int=None, silent=True) -> None:
    if len(self.history) == 0:
      logger.write("Nothing to log.")
      return
    
    latest_epoch_data = self.history[-1]
    epoch = len(self.history) - 1
    # step = None

    log_string = ""

    for name, datum in latest_epoch_data.items():
      # datum here is the array of data point
      # if step == None:
      #   step = len(datum) - 1
      # If it is step, then int should be displayed instead of float number
      if name == PROFILE_NAME.step:
        log_string += "%s: %d; "%(name, datum[-1])
      else:
        log_string += "%s: %.6f; "%(name, datum[-1])
    
    if total == None:
      to_log = f"Epoch {int(epoch)} step {int(step)}: " + log_string[:-2]
    else:
      to_log = f"Epoch {int(epoch)} [{int(step)}/{int(total)}] " + log_string[:-2]

    if silent:
      logger.write_silent(to_log)
    else:
      logger.write(to_log)


def get_param_from_experiment_config(E: Experiment, key: str, alternative_value):
  return E.config[key] if key in E.config else alternative_value

class DannProfile(Profile):
  def __init__(self, history_path: str = None) -> None:
    super().__init__(history_path)

  def plot_loss(self) -> None:
    pass

  def tSNE(self) -> None:
    pass



