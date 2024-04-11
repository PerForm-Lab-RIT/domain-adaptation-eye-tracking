"""
Created on Nov 18, 2022
@author: Viet Nguyen
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import os
from munch import Munch

from ..model.segment import SobelMultiChannel
from ..utils.image import one_hot_label

def get_loss_func_by_loss_config(loss_config: Munch):
    return 1


class FocalLoss2d(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss2d,self).__init__()
        self.gamma = gamma 
        self.loss = nn.NLLLoss(weight)
    def forward(self, outputs, targets):
        return self.loss((1 - nn.Softmax2d()(outputs)).pow(self.gamma) * torch.log(nn.Softmax2d()(outputs)), targets)

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets.long())
    
class SurfaceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []
    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2)*distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2) # Mean between pixels per channel
        score = torch.mean(score, dim=1) # Mean between channels
        return score
    
    
class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, softmax=True, reduction=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = []
        self.reduction = reduction
        if softmax:
            self.norm = nn.Softmax(dim=1)
        else:
            self.norm = nn.Sigmoid()

    def forward(self, ip, target, n_classes=4):
        Label = (np.arange(n_classes) == target.cpu().numpy()[..., None]).astype(np.uint8)
        target = torch.from_numpy(np.rollaxis(Label, 3,start=1)).cuda()
        assert ip.shape == target.shape
        ip = self.norm(ip)

        # Flatten for multidimensional data
        ip = torch.flatten(ip, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        
        numerator = ip*target
        denominator = ip + target

        class_weights = 1./(torch.sum(target, dim=2)**2).clamp(min=self.epsilon)

        A = class_weights*torch.sum(numerator, dim=2)
        B = class_weights*torch.sum(denominator, dim=2)

        dice_metric = 2.*torch.sum(A, dim=1)/torch.sum(B, dim=1)
        if self.reduction:
            return torch.mean(1. - dice_metric.clamp(min=self.epsilon))
        else:
            return 1. - dice_metric.clamp(min=self.epsilon)


class SegLossObject(object):
  def __init__(self, device, dice=True, n_classes=4) -> None:
    self.ce_loss_object = CrossEntropyLoss2d()
    self.gd_loss_object = GeneralizedDiceLoss(softmax=True, reduction=True)
    self.surface_loss_object = SurfaceLoss()
    self.dice = dice
    self.n_classes = n_classes
    self.device = device

  def __call__(self, true, pred, spatial_weights=None, distance_matrix=None, alpha_at_epoch=1, dice=True):
    CE_loss = self.ce_loss_object(pred,true)
    if spatial_weights is not None:
      loss = CE_loss*(torch.from_numpy(np.ones(spatial_weights.shape)).to(torch.float32).to(self.device)+(spatial_weights).to(torch.float32).to(self.device))
    else:
      loss = CE_loss

    loss=torch.mean(loss).to(torch.float32).to(self.device)
    if self.dice and dice:
      loss_dice = self.gd_loss_object(pred,true,n_classes=self.n_classes)
    loss_dice = 0

    if distance_matrix is not None:
      loss_sl = torch.mean(self.surface_loss_object(pred.to(self.device),(distance_matrix).to(self.device)))
    else:
      loss_sl = 0

    loss = (1-alpha_at_epoch)*loss_sl+alpha_at_epoch*(loss_dice)+loss 
    
    return loss

class DannLossObject(object):
  def __init__(self, device) -> None:
    self.seg_loss_object = SegLossObject(device)
    self.device = device

  def __call__(self, true, pred, spatial_weights, distance_matrix, domain_true, domain_pred, alpha_at_epoch):
    seg_loss = self.seg_loss_object(true, pred, spatial_weights, distance_matrix, alpha_at_epoch)
    domain_loss = torch.nn.BCELoss()(domain_pred, domain_true)
    
    return seg_loss + domain_loss, seg_loss, domain_loss


class StructureRetainLoss(object):
  """ Use for SR-cycle gan

  Args:
      object (_type_): _description_
  """
  def __init__(self, gamma_edge=40, gamma_var=2, gamma_mean=0.5, alpha=1, beta=0.01, n_classes=4, edge_threshold=0.4, device=torch.device("cpu")) -> None:
    self.gamma_edge = gamma_edge
    self.gamma_var = gamma_var
    self.gamma_mean = gamma_mean
    self.alpha = alpha
    self.beta = beta
    self.n_classes = n_classes
    self.device = device

    self.edge_threshold = edge_threshold

    # Sobel filters
    self.sobel_img = SobelMultiChannel(1, normalize=True).to(device)
    self.sobel_segment = SobelMultiChannel(n_classes).to(device)


  def __call__(self, cycled_img, orig_img, orig_segment, target_img, target_segment, recov_img):
    # Compute sobel segment map
    segment_label = one_hot_label(orig_segment, n_classes=self.n_classes)
    sobel_segment_map = self.sobel_segment(segment_label)
    sobel_segment_map, indices = sobel_segment_map.max(1, keepdim=True)
    sobel_segment_map[torch.where(sobel_segment_map < self.edge_threshold)] = 0

    # Compute edge of both original and cycled image and recov
    sobel_cycled = self.sobel_img(cycled_img)
    sobel_orig_img = self.sobel_img(orig_img)
    sobel_recov = self.sobel_img(recov_img)
    sobel_cycled[torch.where(sobel_cycled < self.edge_threshold)] = 0
    sobel_orig_img[torch.where(sobel_orig_img < self.edge_threshold)] = 0
    sobel_recov[torch.where(sobel_recov < self.edge_threshold)] = 0

    ### Edge loss for cycled ~ orig
    edge_l_orig = torch.abs(sobel_cycled - sobel_orig_img)
    img_edge_loss_orig = self.alpha * torch.mean(edge_l_orig) + self.beta * torch.mean(edge_l_orig * torch.sign(sobel_segment_map))
    ### Edge loss for cycled ~ recov
    edge_l_recov = torch.abs(sobel_cycled - sobel_recov)
    img_edge_loss_recov = self.alpha * torch.mean(edge_l_recov) + self.beta * torch.mean(edge_l_recov * torch.sign(sobel_segment_map))
    img_edge_loss = self.gamma_edge * (img_edge_loss_orig + img_edge_loss_recov) / 2
    masked_cycled = cycled_img * segment_label

    # Compute segment_label_target
    segment_label_target = one_hot_label(target_segment, n_classes=self.n_classes)
    masked_target = target_img * segment_label_target

    # compute mean manually
    n_val_masked_target = torch.sum(segment_label_target, dim=[2,3], keepdim=True)
    n_val_masked_orig = torch.sum(segment_label, dim=[2,3], keepdim=True)

    mean_target = torch.sum(masked_target, dim=[2,3], keepdim=True) / (n_val_masked_target + 1e-8)
    variance_target = torch.mean(
      torch.pow(
        masked_target - mean_target,
        2
      ) * segment_label_target,
      dim=[2,3]
    )

    mean_cycled = torch.sum(masked_cycled, dim=[2,3], keepdim=True) / (n_val_masked_orig + 1e-8)
    variance_cycled = torch.mean(
      torch.pow(
        masked_cycled - mean_cycled,
        2
      ) * segment_label,
      dim=[2,3]
    )

    variance_loss = self.gamma_var * torch.mean(torch.abs(variance_cycled - variance_target))

    mean_class_loss = self.gamma_mean * torch.mean(torch.abs(mean_cycled - mean_target))

    return img_edge_loss + variance_loss + mean_class_loss