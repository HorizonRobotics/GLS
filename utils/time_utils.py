# Project GLS
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid
import math
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift):
        x = self.norm(x)

        scale, shift = scale_shift
        x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=4, theta = 1000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DeformNetwork(nn.Module):
    def __init__(self, dim=16, num_classes=256, num_images=1600):
        super(DeformNetwork, self).__init__()
        time_dim = dim
        self.classifier = nn.Conv2d(time_dim, num_classes, kernel_size=1)

    def forward(self, objs, t):
        
        logits = self.classifier(objs).squeeze(0)

        return logits