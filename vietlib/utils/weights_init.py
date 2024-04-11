

import torch.nn as nn
import tensorflow as tf
import numpy as np

def fan_in_uniform_init(tensor, fan_in=None):
  """Utility function for initializing actor and critic"""
  # https://github.com/schneimo/ddpg-pytorch/blob/7fb6383627b3b95ee55a36322f7e496aef03872b/utils/nets.py#L12
  if fan_in is None:
    fan_in = tensor.size(-1)

  w = 1. / np.sqrt(fan_in)
  nn.init.uniform_(tensor, -w, w)

def fan_in_uniform_weights_initilization_all(layer):
  # https://stackoverflow.com/a/49433937
  if isinstance(layer, nn.Linear):
    fan_in_uniform_init(layer.weight)
    fan_in_uniform_init(layer.bias)

class FanInUniformInitializer(tf.initializers.Initializer):
  def __init__(self, seed=None):
    self.seed = seed

  def __call__(self, shape, dtype=tf.float32):
    input_dim = shape[0]
    limit = 1 / np.sqrt(input_dim)
    return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed)
