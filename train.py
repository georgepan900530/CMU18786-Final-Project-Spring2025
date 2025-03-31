"""
This file is refactored from the unofficial implementation of the paper (https://github.com/shleecs/DeRaindrop_unofficial)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from options.train_options import TrainOptions
from train_module import trainer
torch.autograd.set_detect_anomaly(True)
opt = TrainOptions().parse()
tr = trainer(opt)
tr.train_start()
