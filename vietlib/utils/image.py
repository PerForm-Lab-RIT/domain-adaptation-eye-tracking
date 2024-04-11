
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import Image
import torch
import torchvision
import h5py

from typing import Optional


def conv2d_size_out(size: int, kernel_size: int=5, stride :int=2) -> int:
  """ Number of Linear input connections depends on output of conv2d layers 
  and therefore the input image size, so compute it.

  Args:
    size (int): _description_
    kernel_size (int, optional): _description_. Defaults to 5.
    stride (int, optional): _description_. Defaults to 2.

  Returns:
    int: _description_
  """
  return (size - (kernel_size - 1) - 1) // stride  + 1

def show_img(img, text_title=None, pytorch=False):
  if pytorch:
    _img = torch.permute(img, [1,2,0]).cpu().detach().numpy()
  else:
    _img = img
  _img = np.squeeze(_img)
  plt.figure(facecolor='white')
  if text_title != None:
    plt.title(text_title)
  plt.axis("off")
  if len(_img.shape) == 2:
    plt.imshow(_img, cmap='gray')
  else:
    plt.imshow(_img)
  plt.show()

def show_imgs(images: List[List[np.ndarray]], figsize: Tuple[int]=(10,10)) -> None:
  height, width = len(images), len(images[0])
  axes = []
  f = plt.figure(figsize=figsize, facecolor='white')
  for i in range(height * width):
    # Debug, plot figure
    axes.append(f.add_subplot(height, width, i + 1))
    # subplot_title=("Subplot"+str(""))
    # axes[-1].set_title(subplot_title)  
    plt.imshow(images[i // width][i % width])
  f.tight_layout()
  plt.show()

def get_eye_horizontal_window(img: np.ndarray, label: np.ndarray, image_size: Tuple[int]=(640, 480)):
  """ Given an image of the eye, get the parts where there is an eye in the dedicated size

  Args:
    img (np.ndarray): _description_
    label (np.ndarray): _description_
    image_size (Tuple[int]): width, height
  """
  h, w = img.shape
  r = np.where(label)[0]
  c = int(0.5*(np.max(r) + np.min(r)))
  if c - 160 < 0:
    top, bot = 0, 320
  elif c + 160 >= h:
    top, bot = h - 320, h
  else:
    top, bot = c-160, c+160

  I = img[top:bot, :]
  LabelMat = label[top:bot, :]
  if image_size is not None:
    I = cv2.resize(I, image_size, interpolation=cv2.INTER_LANCZOS4)
    LabelMat = cv2.resize(LabelMat, image_size, interpolation=cv2.INTER_NEAREST)

  return I, LabelMat

def one_hot_label(batch_img, n_classes=4):
  # classes = torch.unique(img)
  if len(batch_img.shape) == 3:
    X = None
    for _class in range(n_classes):
      x = (batch_img == _class).unsqueeze(1).float()
      if X is None:
        X = x
        
      else:
        X = torch.cat([X, x], dim=1)
    return X
  elif len(batch_img.shape) == 2:
    X = None
    for _class in range(n_classes):
      x = (batch_img == _class).unsqueeze(0).float()
      if X is None:
        X = x
      else:
        X = torch.cat([X, x], dim=0)
    return X

