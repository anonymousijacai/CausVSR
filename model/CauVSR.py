from __future__ import print_function
from __future__ import division

import torch
import math
import torch.nn as nn
from torch import einsum
from math import log
import torch.optim as optim
from torch.optim import lr_scheduler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from lib.Res2Net import res2net101_v1b_26w_4s
from torchvision import models

import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import copy
import cv2

from torch.autograd import Function, Variable
from sklearn.cluster import KMeans, kmeans_plusplus


class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    # @staticmethod
    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print(
                'Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    # @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)


class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    # @staticmethod
    def forward(self, input):
        CWPF = ClassWisePoolFunction(self.num_maps)
        return CWPF(input)



class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x



def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


def deconv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



# ```
# Codes of the Global Category Eclitation Module
#        will  be released after acceptance
#```
class CAE(nn.Module):
    '''

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the emotional categories.
        stage_num (int): The iteration number for EM.
    '''



class ResNetWSL(nn.Module):

    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()

        #backbone
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)
        self.downconv = nn.Sequential(
            nn.Conv2d(2048, num_classes * num_maps, kernel_size=1, stride=1, padding=0, bias=True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(197, 2048, kernel_size=1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(4096,2048, kernel_size=1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1))

        self.GAP = nn.AvgPool2d(14)
        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

        self.GCEM = CAE(512, 8, 3)
        self.fc0 = ConvBNReLU(512 * 4, 512, 3, 1, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_base = nn.Linear(10240, 8)
        self.att_map = None

    # @staticmethod
    def forward(self, x):

        #emotional-stimuli feature representation
        r1, r4, r3, x2 = self.resnet(x)
        r5 = x2 #2048,14,14

        #Causality based Emotional Perception
        x2 = self.fc0(x2)    #14,512,14,14

        #The details of GCEM will be available after accpetance
        x2, mu, z_t = self.GCEM(x2)
        # print(z_t.size()) #14,512,14,14  #14,8,14,14
        idn = x2
        x1 = None
        b, c, h, w = x2.size()
        att_map = z_t
        self.att_map = att_map
        class_att = att_map.view(b, 8, -1, h, w).mean(dim=2)

        # global map, 8 categories
        for i in range(8):
            if i == 0:
                x1 = torch.matmul(idn, class_att[:, i, :, :].view(b, -1, h, w))
            else:
                x2_temp = torch.matmul(idn, class_att[:, i, :, :].view(b, -1, h, w))
                x1 = torch.cat((x1, x2_temp), dim=1)
        dic_global = x1
        dic_global = self.conv4(dic_global)

        x2 = r5 * dic_global #14,2048,14,14
        x2 = torch.cat((x2, r5), dim=1)
        # print(x2.size())  # 14,2048,14,14
        x2 = self.conv4(x2)

        #inputs for causal sentiment map and classifier 2, respetcively
        x = x2
        x_ori = r5

        #causal pseudo sennnntiment map generation
        x = self.downconv(x)
        # print('after downconv x shape is ', x.shape)
        x_conv = x
        x = self.GAP(x)  # x = self.GMP(x)
        # print('after GAP x shape is ', x.shape)
        x = self.spatial_pooling(x)
        # print('after pooling x shape is ', x.shape)
        x = x.view(x.size(0), -1)
        # print('**end x shape is ', x.shape)
        # print('start x_conv shape is ', x_conv.shape)
        x_conv = self.spatial_pooling(x_conv)
        # print('after pooling x_conv shape is ', x_conv.shape)
        x_conv = x_conv * x.view(x.size(0), x.size(1), 1, 1)
        # print('after * with x, x_conv shape is ', x_conv.shape)
        x_conv = self.spatial_pooling2(x_conv)
        # print('after pooling2 x_conv shape is ', x_conv.shape)



        x_conv_copy = x_conv
        # print('start x_conv_copy shape is ', x_conv_copy.size()) #14,1,14,14
        for num in range(0, 2047):
            x_conv_copy = torch.cat((x_conv_copy, x_conv), 1)
            # print('after cat with x_conv, x_conv_copy shape is ', x_conv_copy.shape)
        x_conv_copy = torch.mul(x_conv_copy, x_ori)

        # print('x_ori shape is ', x_ori.shape)
        # print('after mul with x_ori, x_conv_copy shape is ', x_conv_copy.shape)
        x_conv_copy = torch.cat((x_ori, x_conv_copy), 1)
        # print('after cat with x_ori, x_conv_copy shape is ', x_conv_copy.shape)
        x_conv_copy = self.GAP(x_conv_copy)
        # print('after GAP x_conv_copy shape is ', x_conv_copy.shape)
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0), -1)
        # print('after view x_conv_copy shape is ', x_conv_copy.shape)
        x_conv_copy = self.classifier(x_conv_copy)
        # print('**after classifier x_conv_copy shape is ', x_conv_copy.shape)
        # return x, x_conv_copy, x_conv (for visualization)
        return x, x_conv_copy