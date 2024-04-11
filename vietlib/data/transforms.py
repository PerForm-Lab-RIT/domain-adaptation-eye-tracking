
import numpy as np
import torch
from torch.utils.data import Dataset 
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
import copy

from typing import Tuple, Dict, List
from ..utils.image import show_img

transform_sequence = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]) # (input[channel] - mean[channel]) / std[channel] => [0,1] -> [-1, 1]

transform_sequence_without_normalize = transforms.Compose(
    [transforms.ToTensor()]) # (input[channel] - mean[channel]) / std[channel] => [0,1] -> [-1, 1]


class RandomHorizontalFlip(object):
    def __call__(self, img,label):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT),\
                        label.transpose(Image.FLIP_LEFT_RIGHT)
        return img,label

class Starburst_augment_raw_open_eds(object):
    def __call__(self, img):
        x=np.random.randint(1, 40)
        y=np.random.randint(1, 40)
        mode = np.random.randint(0, 2)
        script_dir = os.path.dirname(__file__)
        rel_path = "./starburst_black.png"
        starburst_path = os.path.join(script_dir, rel_path)
        starburst=Image.open(starburst_path).convert("L")
        if mode == 0:
            starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
            starburst = starburst[:, :-x]
        if mode == 1:
            starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
            starburst = starburst[:, x:]

        img[92+y:549+y,0:400]=np.array(img)[92+y:549+y,0:400]*((255-np.array(starburst))/255)+np.array(starburst)
        return Image.fromarray(img)

class Starburst_augment(object):
    def __call__(self, img):
        x=np.random.randint(1, 10)
        y=np.random.randint(1, 30)
        mode = np.random.randint(0, 2)
        script_dir = os.path.dirname(__file__)
        rel_path = "./starburst_black.png"
        starburst_path = os.path.join(script_dir, rel_path)
        # print(starburst_path)
        starburst=Image.open(starburst_path).convert("L")
        starburst = np.asarray(starburst)
        starburst = cv2.resize(starburst, dsize=(320, 280), interpolation=cv2.INTER_CUBIC)
        
        if mode == 0:
          starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
          starburst = starburst[:, :-x]
        if mode == 1:
          starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
          starburst = starburst[:, x:]

        # Adapted code for image 400 x 640
        a1 = np.array(img)[120+y:400+y,40:360]
        a2 = ((255-np.array(starburst))/255)
        b = np.array(starburst)
        img[120+y:400+y,40:360]= a1 * a2 + b

        return Image.fromarray(img)

def getRandomLine(xc, yc, theta):
    x1 = xc - 50*np.random.rand(1)*(1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*np.random.rand(1) + 50)*(1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    # print(f"x1: {x1}")
    return int(x1), int(y1), int(x2), int(y2)

class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value=np.random.randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img,(7,7),sigma_value))

class Translation(object):
    def __call__(self, base,mask):
        factor_h = 2*np.random.randint(1, 20)
        factor_v = 2*np.random.randint(1, 20)
        mode = np.random.randint(0, 4)
#        print (mode,factor_h,factor_v)
        if mode == 0:
            aug_base = np.pad(base, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_base = aug_base[:-factor_v, :]
            aug_mask = aug_mask[:-factor_v, :]
        if mode == 1:
            aug_base = np.pad(base, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_base = aug_base[factor_v:, :]
            aug_mask = aug_mask[factor_v:, :]
        if mode == 2:
            aug_base = np.pad(base, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_base = aug_base[:, :-factor_h]
            aug_mask = aug_mask[:, :-factor_h]
        if mode == 3:
            aug_base = np.pad(base, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_base = aug_base[:, factor_h:]
            aug_mask = aug_mask[:, factor_h:]
        return Image.fromarray(aug_base), Image.fromarray(aug_mask)     
            
class Line_augment(object):
    def __call__(self, base):
        yc, xc = (0.3 + 0.4*np.random.rand(1))*base.shape
        aug_base = copy.deepcopy(base)
        # aug_base = aug_base.astype(np.uint8)
        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       
        
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

# Modified from pytorch original transforms
import random

import numpy as np
import torch
from PIL import Image


class RandomVerticalFlip(object):
  def __call__(self, img):
    if random.random() < 0.5:
      return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


class DeNormalize(object):
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)
    return tensor


class MaskToTensor(object):
  def __call__(self, img):
    return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class FreeScale(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = tuple(reversed(size))  # size: (h, w)
    self.interpolation = interpolation

  def __call__(self, img):
    return img.resize(self.size, self.interpolation)


class FlipChannels(object):
  def __call__(self, img):
    img = np.array(img)[:, :, ::-1]
    return Image.fromarray(img.astype(np.uint8))


class MaskToTensorOneHot(object):
  def __init__(self, num_classes=19):
    self.num_classes=num_classes
  def __call__(self, img):
    return torch.from_numpy( np.eye(self.num_classes+1)[np.array(img, dtype=np.int32)]).long().transpose(0,2)



