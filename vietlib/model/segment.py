#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Viet
"""
from turtle import forward
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class DenseNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)            
        
        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x,x1),dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out)

def conv_bn_relu(n_in, n_out):
  return nn.Sequential(
    nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(n_out),
    nn.ReLU6(inplace=True)
  )

def depth_wise_conv(n_in):
  return nn.Sequential(
    nn.Conv2d(n_in, n_in, kernel_size=3, stride=1, padding=1, groups=n_in, bias=False),
    nn.BatchNorm2d(n_in),
    nn.ReLU6(inplace=True)
  )

def point_wise_conv(n_in, n_out):
  return nn.Sequential(
    nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(n_out),
    nn.ReLU6(inplace=True)
  )

def point_wise_conv_linear(n_in, n_out):
  return nn.Sequential(
    nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(n_out)
  )

class DenseNet2D_down_block_mobile(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block_mobile, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))

        self.conv11 = depth_wise_conv(input_channels)
        self.conv12 = point_wise_conv(input_channels, input_channels + output_channels)
        # print(f"[DenseNet2D_down_block_mobile] in: {input_channels}, out: {output_channels}")

        self.conv21 = depth_wise_conv(input_channels*2 + output_channels)
        self.conv22 = point_wise_conv(input_channels*2 + output_channels, output_channels)
        self.conv3 = nn.Conv2d(input_channels*2 + output_channels*2, output_channels,kernel_size=(3,3),padding=(1,1))

        self.max_pool = nn.AvgPool2d(kernel_size=down_size)            
        
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x12 = self.dropout1(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.dropout2(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.dropout3(self.conv3(x31))
        else:
            x12 = self.conv12(self.conv11(x))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.conv22(self.conv21(x21))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.conv3(x31)
        return self.bn(out)
    
class DenseNet2D_down_block_mobile_2(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,dropout=False,prob=0):
        super(DenseNet2D_down_block_mobile_2, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))

        self.conv11 = depth_wise_conv(input_channels)
        self.conv12 = point_wise_conv(input_channels, output_channels)
        # print(f"[DenseNet2D_down_block_mobile] in: {input_channels}, out: {output_channels}")

        self.conv21 = depth_wise_conv(input_channels + output_channels)
        self.conv22 = point_wise_conv(input_channels + output_channels, output_channels)
        self.conv3 = nn.Conv2d(input_channels + output_channels*2, output_channels, kernel_size=(3,3),padding=(1,1))

        self.max_pool = nn.AvgPool2d(kernel_size=down_size)
        
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
            
        if self.dropout:
            x12 = self.dropout1(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.dropout2(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.dropout3(self.conv3(x31))
        else:
            x12 = self.conv12(self.conv11(x))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.conv22(self.conv21(x21))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.conv3(x31)
        return self.bn(out)

class MobileNetBlock(nn.Module):
    def __init__(self,input_channels,output_channels,dropout=False,prob=0):
        super().__init__()
        # self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))

        self.conv11 = depth_wise_conv(input_channels)
        self.conv12 = point_wise_conv(input_channels, output_channels)
        # print(f"[DenseNet2D_down_block_mobile] in: {input_channels}, out: {output_channels}")

        self.conv21 = depth_wise_conv(input_channels + output_channels)
        self.conv22 = point_wise_conv(input_channels + output_channels, output_channels)
        self.conv3 = nn.Conv2d(input_channels + output_channels*2, output_channels, kernel_size=(3,3),padding=(1,1))

        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
    
    def forward(self, x):
        if self.dropout:
            x12 = self.dropout1(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.dropout2(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.dropout3(self.conv3(x31))
        else:
            x12 = self.conv12(self.conv11(x))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.conv22(self.conv21(x21))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.conv3(x31)
        return self.bn(out)

class MobileNetDownBlock(MobileNetBlock):
    def __init__(self, input_channels, output_channels, down_size: int, dropout=False, prob=0):
        super().__init__(input_channels, output_channels, dropout, prob)
        self.max_pool = nn.MaxPool2d(kernel_size=down_size)
        self.down_size = down_size

    def forward(self, x):
        if self.down_size != None:
            x = self.max_pool(x)
        if self.dropout:
            x12 = self.dropout1(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.dropout2(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.dropout3(self.conv3(x31))
        else:
            x12 = self.conv12(self.conv11(x))
            x21 = torch.cat((x,x12),dim=1)
            x22 = self.conv22(self.conv21(x21))
            x31 = torch.cat((x21,x22),dim=1)
            out = self.conv3(x31)
        return self.bn(out)

class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
        return out

class DenseNet2D_up_block_concat_mobile(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super(DenseNet2D_up_block_concat_mobile, self).__init__()

        self.conv11 = depth_wise_conv(skip_channels + input_channels)
        self.conv12 = point_wise_conv(skip_channels + input_channels, output_channels)

        self.conv21 = depth_wise_conv(skip_channels + input_channels + output_channels)
        self.conv22 = point_wise_conv(skip_channels + input_channels + output_channels, output_channels)

        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.dropout1(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.dropout2(self.conv22(self.conv21(x21)))
        else:
            x1 = self.conv12(self.conv11(x))
            x21 = torch.cat((x,x1),dim=1)
            out = self.conv22(self.conv21(x21))
        return out

class MobileNetUpBlockWithConcat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,dropout=False,prob=0):
        super().__init__()

        self.conv11 = depth_wise_conv(skip_channels + input_channels)
        self.conv12 = point_wise_conv(skip_channels + input_channels, output_channels)

        self.conv21 = depth_wise_conv(skip_channels + input_channels + output_channels)
        self.conv22 = point_wise_conv(skip_channels + input_channels + output_channels, output_channels)

        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self,prev_feature_map,x):
        x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')
        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            x1 = self.dropout1(self.conv12(self.conv11(x)))
            x21 = torch.cat((x,x1),dim=1)
            out = self.dropout2(self.conv22(self.conv21(x21)))
        else:
            x1 = self.conv12(self.conv11(x))
            x21 = torch.cat((x,x1),dim=1)
            out = self.conv22(self.conv21(x21))
        return out

class DenseNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=32,concat=True,dropout=False,prob=0):
        super(DenseNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out



class DenseMobileNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=16,concat=True,dropout=False,prob=0):
        super(DenseMobileNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block_mobile(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block_mobile(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block_mobile(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block_mobile(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block_mobile(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out

class FeatureEncoder(nn.Module):
  def __init__(self,in_channels=1,channel_size=32,out_channels=256, dropout=False,prob=0) -> None:
    super().__init__()

    self.down_block1 = MobileNetDownBlock(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
    self.down_block2 = MobileNetDownBlock(input_channels=channel_size,output_channels=channel_size,
                                              down_size=(2,2),dropout=dropout,prob=prob)
    self.down_block3 = MobileNetDownBlock(input_channels=channel_size,output_channels=channel_size,
                                              down_size=(2,2),dropout=dropout,prob=prob)
    self.down_block4 = MobileNetDownBlock(input_channels=channel_size,output_channels=channel_size,
                                              down_size=(2,2),dropout=dropout,prob=prob)
    self.down_block5 = MobileNetDownBlock(input_channels=channel_size,output_channels=out_channels,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

  def forward(self, x):
    tensor = self.down_block1(x)
    tensor = self.down_block2(tensor)
    tensor = self.down_block3(tensor)
    tensor = self.down_block4(tensor)
    tensor = self.down_block5(tensor)
    return tensor

class FeatureExtractor(nn.Module):
  """ Return the 1 feature latent encoding, with 4 multiple layer output,
  x1, x2, x3, x4 is the decreasing feature, with x5 is the last decreasing, which is also the laten encoding.
  """
  def __init__(self,in_channels=1,out_channels=32, channel_size=32, dropout=False,prob=0) -> None:
    super().__init__()

    self.down_block1 = MobileNetDownBlock(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
    self.down_block2 = MobileNetDownBlock(input_channels=channel_size,output_channels=channel_size,
                                              down_size=(2,2),dropout=dropout,prob=prob)
    self.down_block3 = MobileNetDownBlock(input_channels=channel_size,output_channels=channel_size,
                                              down_size=(2,2),dropout=dropout,prob=prob)
    self.down_block4 = MobileNetDownBlock(input_channels=channel_size,output_channels=channel_size,
                                              down_size=(2,2),dropout=dropout,prob=prob)
    self.down_block5 = MobileNetDownBlock(input_channels=channel_size,output_channels=out_channels,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

    self._initialize_weights()

  def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

  def forward(self, x, return_encodings=True):
    x1 = self.down_block1(x)
    x2 = self.down_block2(x1)
    x3 = self.down_block3(x2)
    x4 = self.down_block4(x3)
    x5 = self.down_block5(x4) # (B, 32, 15, 20)
    if return_encodings:
        return x1, x2, x3, x4, x5
    else:
        return x5

class FeatureReconstructor(nn.Module):
  """ Takes in multiple layer encoding and perform 4 upblock concat

  Args:
      nn (_type_): _description_
  """
  def __init__(self, in_channels=32,out_channels=4,channel_size=32,concat=True,dropout=False,prob=0) -> None:
    super().__init__()
    self.up_block1 = MobileNetUpBlockWithConcat(skip_channels=channel_size,input_channels=in_channels,
                                                  output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
    self.up_block2 = MobileNetUpBlockWithConcat(skip_channels=channel_size,input_channels=channel_size,
                                                output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
    self.up_block3 = MobileNetUpBlockWithConcat(skip_channels=channel_size,input_channels=channel_size,
                                                output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
    self.up_block4 = MobileNetUpBlockWithConcat(skip_channels=channel_size,input_channels=channel_size,
                                                output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

    self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
    self.concat = concat
    self.dropout = dropout
    self.dropout1 = nn.Dropout(p=prob)
    
    self._initialize_weights()

  def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
  def forward(self, x1, x2, x3, x4, x5):
      # x5: (B, 32, 15, 20)

      x6 = self.up_block1(x4,x5)
      x7 = self.up_block2(x3,x6)
      x8 = self.up_block3(x2,x7)
      x9 = self.up_block4(x1,x8)
      if self.dropout:
          out = self.out_conv1(self.dropout1(x9))
      else:
          out = self.out_conv1(x9)
                      
      return out

class ImageReconstructor(FeatureReconstructor):
  """ Basically it is the same as feature reconstructor with additional
  last layer as Tanh so that it outputs the image range. 9 resblock

  """

  def __init__(self, in_channels=32, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0) -> None:
    super().__init__(in_channels, out_channels, channel_size, concat, dropout, prob)
    self.res1 = ResidualBlock(in_channels)
    self.res2 = ResidualBlock(in_channels)
    self.res3 = ResidualBlock(in_channels)
    self.res4 = ResidualBlock(in_channels)
    self.res5 = ResidualBlock(in_channels)
    self.res6 = ResidualBlock(in_channels)
    self.res7 = ResidualBlock(in_channels)
    self.res8 = ResidualBlock(in_channels)
    self.res9 = ResidualBlock(in_channels)
    self.tanh = nn.Tanh()

  def forward(self, *args):
    x1, x2, x3, x4, x5 = args
    _x5 = self.res9(self.res8(self.res7(self.res6(self.res5(self.res4(self.res3(self.res2(self.res1(x5)))))))))
    return self.tanh(super().forward(x1, x2, x3, x4, _x5))

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

class Discriminator(nn.Module):
  def __init__(self, input_shape):
    super(Discriminator, self).__init__()
    
    channels, height, width = input_shape
    
    # Calculate output shape of image discriminator (PatchGAN)
    self.output_shape = (1, height//2**4, width//2**4)
    
    def discriminator_block(in_filters, out_filters, normalize=True):
      """Returns downsampling layers of each discriminator block"""
      layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
      if normalize:
          layers.append(nn.InstanceNorm2d(out_filters))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers
    
    self.model = nn.Sequential(
      *discriminator_block(channels, 32, normalize=False),
      *discriminator_block(32, 64),
      *discriminator_block(64,128),
      *discriminator_block(128,256),
      nn.ZeroPad2d((1,0,1,0)),
      nn.Conv2d(256, 1, 4, padding=1)
    )

  def forward(self, img):
    return self.model(img)

class StateFeatureExtractor(nn.Module):
  def __init__(self, input_channels, channel_size=32, output_channels=256, dropout=False, prob=0) -> None:
    """ 

    Args:
        state_shape (_type_): _description_
        output_features (_type_): _description_
    """
    super().__init__()
    
    self.mob1 = MobileNetBlock(
      input_channels=input_channels,
      output_channels=channel_size,
      dropout=dropout,
      prob=prob
    )

    self.mob2 = MobileNetBlock(
      input_channels=channel_size,
      output_channels=channel_size,
      dropout=dropout,
      prob=prob
    )

    self.mob3 = MobileNetBlock(
      input_channels=channel_size,
      output_channels=channel_size,
      dropout=dropout,
      prob=prob
    )

    self.mob4 = MobileNetBlock(
      input_channels=channel_size,
      output_channels=output_channels,
      dropout=dropout,
      prob=prob
    )
  
  def forward(self, x):
    tensor = self.mob1(x)
    tensor = self.mob2(tensor)
    tensor = self.mob3(tensor)
    tensor = self.mob4(tensor)
    return tensor

class DenseMobileNet2D_2(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=16,concat=True,dropout=False,prob=0):
        super(DenseMobileNet2D_2, self).__init__()

        self.down_block1 = DenseNet2D_down_block_mobile_2(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.up_block1(x4,x5)
        x7 = self.up_block2(x3,x6)
        x8 = self.up_block3(x2,x7)
        x9 = self.up_block4(x1,x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(x9))
        else:
            out = self.out_conv1(x9)
                       
        return out

class RegressionModule(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        inChannels = in_channel
        self.max_pool = nn.AvgPool2d(kernel_size=2)

        self.c1 = nn.Conv2d(in_channels=inChannels,
                            out_channels=64,
                            bias=True,
                            kernel_size=3,
                            padding=1)

        self.c2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            bias=True,
                            kernel_size=3,
                            padding=1)

        self.c3 = nn.Conv2d(in_channels=64+inChannels,
                            out_channels=32,
                            kernel_size=1,
                            bias=True)

        self.l1 = nn.Linear(32, 128, bias=True)
        self.l2 = nn.Linear(128, 10, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=inChannels)
        self.bn2 = nn.BatchNorm2d(num_features=64+inChannels)

        self.c_actfunc = F.hardtanh # Center has to be between -1 and 1
        self.param_actfunc = F.hardtanh # Parameters can't be negative and capped to 1

    def forward(self, x, alpha):
        # x: [B, C, 15, 20]
        B = x.shape[0]
        p = self.bn1(x)
        p = self.c1(p)
        p = self.c2(p)
        x = torch.cat([x, p], dim=1)
        x = self.bn2(x)
        x = self.c3(x)
        x = self.l1(x.reshape(B, 32, -1).sum(dim=-1))
        x = self.l2(x)

        EPS = 1e-5

        pup_c = self.c_actfunc(x[:, 0:2], min_val=-1+EPS, max_val=1-EPS)
        pup_param = self.param_actfunc(x[:, 2:4], min_val=0+EPS, max_val=1-EPS)
        pup_angle = x[:, 4]
        iri_c = self.c_actfunc(x[:, 5:7], min_val=-1+EPS, max_val=1-EPS)
        iri_param = self.param_actfunc(x[:, 7:9], min_val=0+EPS, max_val=1-EPS)
        iri_angle = x[:, 9]


        op = torch.cat([pup_c,
                        pup_param,
                        pup_angle.unsqueeze(1),
                        iri_c,
                        iri_param,
                        iri_angle.unsqueeze(1)], dim=1)
        return op


class Ellseg(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=16,concat=True,dropout=False,prob=0):
        super(Ellseg, self).__init__()

        self.down_block1 = DenseNet2D_down_block_mobile_2(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.linear_module = RegressionModule(channel_size)

        self.up_block1 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x, alpha):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)

        latent = torch.mean(self.x5.flatten(start_dim=2), -1) # [B, features]
        elOut = self.elReg(x, alpha) # Linear regression to ellipse parameters

        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out, elOut, latent


from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output.dtype)
        output = -1 * grad_output

        return output * ctx.alpha, None


class DomainClassifier(nn.Module):
    def __init__(self, input_length) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_length, 512)
        # self.bn1 = nn.BatchNorm1d(512)

        self.lin2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)

        self.lin3 = nn.Linear(256, 128)
        # self.bn3 = nn.BatchNorm1d(128)

        self.lin4 = nn.Linear(128, 64)
        # self.bn4 = nn.BatchNorm1d(64)

        self.lin5 = nn.Linear(64, 1)
    
    def forward(self, x):
        tensor = self.lin1(x)
        # tensor = self.bn1(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin2(tensor)
        # tensor = self.bn2(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin3(tensor)
        # tensor = self.bn3(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin4(tensor)
        # tensor = self.bn4(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin5(tensor)
        out = torch.sigmoid(tensor)

        return out

class DomainClassifier_v2(nn.Module):
    def __init__(self, in_channels, height, width) -> None:
        super().__init__()

        self.down_block_1 = MobileNetDownBlock(in_channels, 16, (2,2), True, 0.2)

        self.lin1 = nn.Linear(math.ceil(height / 2) * math.ceil(width / 2) * 16, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.lin2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(256)

        self.lin3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(128)

        self.lin4 = nn.Linear(32, 1)
    
    def forward(self, x5):
        # x5 has the shape (B, 32, 15, 20)
        tensor = self.down_block_1(x5)
        tensor = self.lin1(tensor)
        tensor = self.bn1(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin2(tensor)
        tensor = self.bn2(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin3(tensor)
        tensor = self.bn3(tensor)
        tensor = F.relu(tensor)

        tensor = self.lin4(tensor)
        out = torch.sigmoid(tensor)

        return out

class GANGenrator(nn.Module):
  def __init__(self, in_channels, out_channels, channel_size) -> None:
    super().__init__()
    
    self.encoder = FeatureExtractor(in_channels, 32, channel_size, True, 0.2)
    self.decoder = ImageReconstructor(32, out_channels, channel_size, True, True, 0.2)
  
  def forward(self, x):
    return self.decoder(*self.encoder(x))

from torch import nn
import torch


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return torch.clamp(x, 0, 1)

class SobelMultiChannel(nn.Module):
    def __init__(self, n_channels, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.filterX = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=n_channels)
        self.filterY = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=n_channels)
        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).unsqueeze(0).unsqueeze(0).broadcast_to((n_channels, 1, 3, 3))
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).unsqueeze(0).unsqueeze(0).broadcast_to((n_channels, 1, 3, 3))
        self.filterX.weight = nn.Parameter(Gx, requires_grad=False)
        self.filterY.weight = nn.Parameter(Gy, requires_grad=False)

    def forward(self, img):
        if self.normalize:
            _img = img / 2 + 0.5
        else:
            _img = img
        x = self.filterX(_img)
        y = self.filterY(_img)
        x2 = torch.mul(x, x)
        y2 = torch.mul(y, y)
        sum_sq = torch.add(x2, y2)
        _sqrt = torch.sqrt(sum_sq + 1e-8)
        return torch.clamp(_sqrt, 0, 1)


class Dann(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=16,concat=True,dropout=False,prob=0):
        super().__init__()

        self.down_block1 = DenseNet2D_down_block_mobile_2(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block_mobile_2(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_mobile(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self.linear_module = RegressionModule(channel_size)
        self.domain_classifier = DomainClassifier(channel_size)

        # self.rev_grad = ReverseLayerF.apply(latent, backward_alpha)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x, alpha, backward_alpha):
        """_summary_

        Args:
            x (_type_): _description_
            alpha (_type_): _description_
            backward_alpha (_type_): _description_

        Returns:
            out,  (B, C, H, W)
            domain_out, (B, 1)
            latent, (B, features)
            elOut, (B, 10)
            flattened (B, features): latent + Reverse Grad
        """
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)


        latent = torch.mean(self.x5.flatten(start_dim=2), -1) # [B, features] # TODO: This part is naive?
        elOut = self.linear_module(self.x5, alpha) # Linear regression to ellipse parameters
        
        flattened = ReverseLayerF.apply(latent, backward_alpha)
        domain_out = self.domain_classifier(flattened)

        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out, domain_out, latent, elOut, flattened


class DannOriginal(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=16,concat=True,dropout=False,prob=0):
        super().__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,output_channels=channel_size,
                                                 down_size=None,dropout=dropout,prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size,
                                                 down_size=(2,2),dropout=dropout,prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self.linear_module = RegressionModule(channel_size)
        self.domain_classifier = DomainClassifier(channel_size)

        # self.rev_grad = ReverseLayerF.apply(latent, backward_alpha)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self,x, alpha, backward_alpha):
        """_summary_

        Args:
            x (_type_): _description_
            alpha (_type_): _description_
            backward_alpha (_type_): _description_

        Returns:
            out,  (B, C, H, W)
            domain_out, (B, 1)
            latent, (B, features)
            elOut, (B, 10)
            flattened (B, features): latent + Reverse Grad
        """
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)


        latent = torch.mean(self.x5.flatten(start_dim=2), -1) # [B, features] # TODO: This part is naive?
        elOut = self.linear_module(self.x5, alpha) # Linear regression to ellipse parameters
        
        flattened = ReverseLayerF.apply(latent, backward_alpha)
        domain_out = self.domain_classifier(flattened)

        self.x6 = self.up_block1(self.x4,self.x5)
        self.x7 = self.up_block2(self.x3,self.x6)
        self.x8 = self.up_block3(self.x2,self.x7)
        self.x9 = self.up_block4(self.x1,self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)
                       
        return out, domain_out, latent, elOut, flattened
