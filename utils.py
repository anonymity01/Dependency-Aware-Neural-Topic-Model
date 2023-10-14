#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
# from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
def prune(A):
  zero = torch.zeros_like(A).to(device)
  A = torch.where(A < 0.3, zero, A)
  return A
def gumble_dag_loss(A):
    expm_A = torch.exp(F.gumbel_softmax(A))
    l = torch.trace(expm_A)-A.size()[0]
    return l
def filldiag_zero(A):
    mask = torch.eye(A.size()[0], A.size()[0]).byte().to(device)
    A.masked_fill_(mask, 0)
    return A
    
def mask_threshold(x):
  x = (x+0.5).int().float()
  return x
def matrix_poly(matrix, d):
    x = torch.eye(d).to(device)+ torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)
     
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A
    
    
def get_parse_args():
    # parse some given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--every_degree', '-N', type=int, default=10,
                        help='every N degree as a partition of dataset')
    args = parser.parse_args()
    return args

def weights_init(m):
    if (type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif (type(m) == nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)   