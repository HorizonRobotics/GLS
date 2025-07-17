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

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        self.data_dic = {}
        for i in range(len(data_names)):
            features = np.load(data_names[i])
            name = data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0] 
            if i == 0:
                data = features
            else:
                data = np.concatenate([data, features], axis=0)
        self.data = data

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 