#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Creator: Viet Dung Nguyen
@Date: April 14, 2022
@Credits: Viet Dung Nguyen
@Version: 0.0.1

"""

from functools import wraps
from . import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import cv2
from PIL import Image
import time
import sys

from typing import Optional
import torch

from torchmetrics.functional import jaccard_index

def timing(f):
  # https://stackoverflow.com/a/27737385
  @wraps(f)
  def wrap(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    print('func:%r took: %2.4f sec' % \
      (f.__name__, te-ts))
    return result
  return wrap

def preprocess_image(img: np.array, image_size=None) -> tf.Tensor:
  """_summary_

  Args:
    img (np.array): Expect image with shape (height, width, 3). With range [0, 255], int
    image_size (tuple, optional): Size of the image (height, width) (does not include the number of channel).
      Defaults to None.

  Returns:
    tf.Tensor: Tensor representation of the preprocessed image.
      Has shape (*image_size, img.shape[2]). Range [0, 1]
  """
  # Preprocessing image
  preprocessed_img = tf.cast(img, tf.float32)
  preprocessed_img /= 255.0
  assert tf.reduce_max(preprocessed_img) <= 1 and tf.reduce_min(preprocessed_img) >= 0, "Wrong behavior"
  if image_size != None:
    preprocessed_img = tf.image.resize(preprocessed_img, image_size, method=tf.image.ResizeMethod.BILINEAR)
  return preprocessed_img

def get_mask_by_layer(img: tf.Tensor, layer: int) -> tf.Tensor:
  """Utilitiy method for preprocess label

  Args:
    img (tf.Tensor): _description_
    layer (int): _description_

  Returns:
    tf.Tensor: _description_
  """
  return tf.cast(
    tf.math.argmax(img, axis=-1) == layer, # Get only the higest value layer
    tf.float32
  )[..., tf.newaxis] # expand last dimension

def preprocess_label(img: np.array, image_size=None) -> tf.Tensor:
  """_summary_

  Args:
    img (np.array): A numpy representation of the image. Expected range uint8 [0-255]

  Returns:
    tf.Tensor: Tensor representation of label
  """
  resized_img = img
  if image_size != None:
    resized_img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)

  eye_mask = tf.cast(
    tf.reduce_any(tf.cast(resized_img, tf.bool), axis=-1),
    tf.float32
  )[..., tf.newaxis]

  background_mask = 1 - eye_mask

  # Masking the raw mask with the eye mask for each pupil, iris, and sclera
  pupil_mask = eye_mask * get_mask_by_layer(resized_img, 2) 

  iris_mask = eye_mask * get_mask_by_layer(resized_img, 1)

  sclera_mask = eye_mask * get_mask_by_layer(resized_img, 0)

  result = tf.concat(
    [background_mask, pupil_mask, iris_mask, sclera_mask],
    axis=-1
  )
  
  return result

def preprocess_label_sparse(img: np.array, image_size=None) -> tf.Tensor:
  """_summary_

  Args:
    img (np.array): A numpy representation of the image. Expected range uint8 [0-255]

  Returns:
    tf.Tensor: Tensor representation of label
  """
  resized_img = img
  if image_size != None:
    resized_img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)

  eye_mask = tf.cast(
    tf.reduce_any(tf.cast(resized_img, tf.bool), axis=-1),
    tf.float32
  )[..., tf.newaxis]

  background_mask = 1 - eye_mask

  # Masking the raw mask with the eye mask for each pupil, iris, and sclera
  pupil_mask = eye_mask * get_mask_by_layer(resized_img, 2) * 3

  iris_mask = eye_mask * get_mask_by_layer(resized_img, 1) * 2

  sclera_mask = eye_mask * get_mask_by_layer(resized_img, 0) * 1

  result = tf.concat(
    [background_mask, pupil_mask, iris_mask, sclera_mask],
    axis=-1
  )
  
  result = tf.reduce_sum(result, axis=-1)
  
  return result

def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: Optional[bool] = True) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (Optional[bool]): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width)
        ys = torch.linspace(-1, 1, height)
    else:
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys])).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def preprocess_label_sparse_np(img: np.array, size=(0, 0)) -> np.ndarray:
  """_summary_

  Args:
    img (np.array): A numpy representation of the image. Expected range uint8 [0-255]

  Returns:
    np.ndarray: (H, W). With 0 is background, 1 is pupil, 2 is iris, and 3 is sclera
  """
  # TODO: Implement resizing label
  eye_mask = np.any(img.astype(bool), axis=-1).astype(float)[..., np.newaxis] # 0 on all bakcground

  masks = np.argmax(img, axis=-1).astype(float)[..., np.newaxis] 

  # Masking the raw mask with the eye mask for each pupil, iris, and sclera
  pupil_mask = eye_mask * (masks == 0) * 3 # 3 on all pupil
  iris_mask = eye_mask * (masks == 1) * 2 # 2 on all iris
  sclera_mask = eye_mask * (masks == 2) * 1 # 1 on all sclera

  result = np.concatenate(
    [pupil_mask, iris_mask, sclera_mask],
    axis=-1
  )
  
  result = np.sum(result, axis=-1)
  
  return result

def preprocess_chengyi_label_sparse_np(img: np.array, size=(0, 0)) -> np.ndarray:
  """ same as preprocess_label_sparse_np, but now process label according to chengyi's labeling order
  In his labelling: channel 0 is pupil, channel 1 is iris, channel 2 is sclera, and channel 3 is background
  in s-general dataset, there is no such background channel

  Args:
    img (np.array): A numpy representation of the image. Expected range uint8 [0-255]

  Returns:
    np.ndarray: (H, W). With 0 is background, 1 is sclera, 2 is iris, and 3 is pupil
  """
  eye_mask = np.any(img[..., :-1] > 128, axis=-1).astype(float)[..., np.newaxis] # 0 on all bakcground
  masks = np.argmax(img[..., :-1], axis=-1).astype(float)[..., np.newaxis] 

  # Masking the raw mask with the eye mask for each pupil, iris, and sclera
  pupil_mask = eye_mask * (masks == 0) * 3 # 3 on all pupil
  iris_mask = eye_mask * (masks == 1) * 2 # 2 on all iris
  sclera_mask = eye_mask * (masks == 2) * 1 # 1 on all sclera

  result = np.concatenate(
    [pupil_mask, iris_mask, sclera_mask],
    axis=-1
  )
  
  result = np.sum(result, axis=-1)
  
  return result.astype(int)


"""
Created on Tue Aug 27 16:04:18 2019

@author: Aayush Chaudhary

References:
    https://evalai-forum.cloudcv.org/t/fyi-on-semantic-segmentation/180
    https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py
    https://discuss.pytorch.org/t/using-cross-entropy-loss-with-semantic-segmentation-model/31988
    https://github.com/LIVIAETS/surface-loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import os

from sklearn.metrics import precision_score , recall_score,f1_score
from scipy.ndimage import distance_transform_edt as distance


#https://github.com/LIVIAETS/surface-loss
def one_hot2dist(posmask):
    # Input: Mask. Will be converted to Bool.
    # Author: Rakshit Kothari
    assert len(posmask.shape) == 2
    h, w = posmask.shape
    res = np.zeros_like(posmask)
    posmask = posmask.astype(bool)
    mxDist = np.sqrt((h-1)**2 + (w-1)**2)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res/mxDist

def mIoU(predictions, targets,info=False):  ###Mean per class accuracy
    unique_labels = np.unique(targets)
    num_unique_labels = len(unique_labels)
    ious = []
    for index in range(num_unique_labels):
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection.numpy())/np.sum(union.numpy())
        ious.append(iou_score)
    if info:
        print ("per-class mIOU: ", ious)
    return np.mean(ious)

def compute_mean_iou(flat_pred, flat_label,info=False):
    '''
    compute mean intersection over union (IOU) over all classes
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return: mean IOU
    '''
    unique_labels = np.unique(flat_label)
    num_unique_labels = len(unique_labels)

    Intersect = np.zeros(num_unique_labels)
    Union = np.zeros(num_unique_labels)
    precision = np.zeros(num_unique_labels)
    recall = np.zeros(num_unique_labels)
    f1 = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = flat_pred == val
        label_i = flat_label == val
        
        if info:
            precision[index] = precision_score(pred_i, label_i, 'weighted')
            recall[index] = recall_score(pred_i, label_i, 'weighted')
            f1[index] = f1_score(pred_i, label_i, 'weighted')
        
        Intersect[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        Union[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    if info:
        print ("per-class mIOU: ", Intersect / Union)
        print ("per-class precision: ", precision)
        print ("per-class recall: ", recall)
        print ("per-class f1: ", f1)
    mean_iou = np.mean(Intersect / Union)
    return mean_iou


def mIoU_v2(pred, true, n_unique_labels=None) -> float:
  """
  this uses deprecated torchmetrics version
  Docs on this site: https://torchmetrics.readthedocs.io/en/v0.10.3/classification/jaccard_index.html?highlight=jaccard%20index
  Args:
    pred (_type_): can be (B, C, H, W) probs map or (B, H, W)
    true (_type_): Must be (B, H, W) for class label

  Returns:
    float: MIOU
  """
  predicts = pred
  label = true
  # if not isinstance(pred, torch.Tensor):
  #   predicts = torch.Tensor(pred.numpy()).int()
  # if not isinstance(true, torch.Tensor):
  #   label = torch.Tensor(true.numpy()).int()
  if n_unique_labels is None:
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
  else:
    num_unique_labels = n_unique_labels
  
  miou = jaccard_index(predicts, label, num_classes=num_unique_labels)
  # jaccard_index(output.cpu(), tensor_label, num_classes=num_unique_labels)
  return miou.item()

def mIoU_segment(pred, true, n_unique_labels=None) -> float:
  """
  this uses deprecated torchmetrics version
  Docs on this site: https://torchmetrics.readthedocs.io/en/v0.10.3/classification/jaccard_index.html?highlight=jaccard%20index
    pred (_type_): can be (B, C, H, W) probs map or (B, H, W)
    true (_type_): Must be (B, H, W) for class label

  Returns:
    float: MIOU
  """
  predicts = pred
  label = true
  if n_unique_labels is None:
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
  else:
    num_unique_labels = n_unique_labels
  
  miou = jaccard_index(predicts, label, num_classes=num_unique_labels, average="none")
  return miou

def total_metric(nparams,miou):
    S = nparams * 4.0 /  (1024 * 1024)
    total = min(1,1.0/S) + miou
    return total * 0.5
    
    
def get_nparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_predictions(output):
    bs,c,h,w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs,h,w) # bs x h x w
    return indices


###########################

# Method is not working correctly
def free_memories(to_delete: List[str]):
  """ Given a list of string of variables, Free them
  https://stackoverflow.com/a/64594787

  Args:
    to_delete (List[str]): _description_
  """
  for _var in to_delete:
    if _var in locals() or _var in globals():
      exec(f'del {_var}')


def time_record(func):
  """ https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk

  Args:
      func (_type_): _description_

  Returns:
      _type_: _description_
  """
  @wraps(func)
  def timeit_wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
    return result
  return timeit_wrapper


def get_here(name: str=""):
  print(f"Get Here: {name}")

def Override(f):
  return f

def confusion_matrix_pytorch(cm, output_flatten, target_flatten, num_classes):
  for i in range(num_classes):
    for j in range(num_classes):
      cm[i, j] = cm[i, j] + ((output_flatten == i) * (target_flatten == j)).sum().type(torch.IntTensor).cuda()
  return cm


# https://www.kaggle.com/code/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/script
# This method in pytorch compute iou across each item in the batch
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

  SMOOTH = 1e-6

  # If it is the probability map, then the
  if len(outputs.shape) == 4:
    outputs = get_predictions(outputs).cpu()
  labels = labels.cpu()

  intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
  union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
  
  iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
  
  thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
  
  return thresholded  # Or thresholded.mean() if you are interested in average across the batch


def check_vscode_interactive() -> bool:
  return hasattr(sys, 'ps1')


def plot_histogram_2ds(x: tf.Tensor, y: tf.Tensor, figpath=None):
  # plotting the next 16 steps in the plan
  # x.shape == y.shape: (planning_horizon, n_candidates)
  B = x.shape[0]
  fig = plt.figure(figsize=(10, 10))
  for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.hist2d(x[i, :], y[i, :], bins=[50, 50])
    plt.colorbar()
  plt.tight_layout()
  if figpath is not None:
    plt.savefig(figpath)
  return fig

def plot_scatters(x: tf.Tensor, y: tf.Tensor, figpath=None):
  # plotting the next 16 steps in the plan
  # x.shape == y.shape: (planning_horizon, n_candidates)
  B = x.shape[0]
  fig = plt.figure(figsize=(10, 10))
  for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.scatter(x[i, ...], y[i, ...], s=2, alpha=0.4)
  plt.tight_layout()
  if figpath is not None:
    plt.savefig(figpath)
  return fig