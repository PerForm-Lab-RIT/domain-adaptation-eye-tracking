

import torch.nn as nn

class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()
    
    self.block = nn.Sequential(
      nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features), 
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3),
      nn.InstanceNorm2d(in_features)
    )

  def forward(self, x):
    return x + self.block(x)


class SimpleResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()
    
    self.block = nn.Sequential(
      nn.ReflectionPad2d(1), # Pads the input tensor using the reflection of the input boundary
      nn.Conv2d(in_features, in_features, 3),
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_features, in_features, 3)
    )

  def forward(self, x):
    return x + self.block(x)