# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from .modules import DSConv

# Tools lib
import numpy as np
import cv2
import random
import time
import os


# Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=1, kernel_size=5, stride=1, padding=2
            )
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=5, stride=4, padding=1
            ),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=5, stride=4, padding=1
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 1024), nn.Linear(1024, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        return mask, self.fc(x)


# DCN Discriminator
class DSConvDiscriminator(nn.Module):
    def __init__(self):
        super(DSConvDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            DSConv(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            DSConv(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            DSConv(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            DSConv(
                in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            DSConv(
                in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            DSConv(
                in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
        )
        self.conv_mask = nn.Sequential(
            DSConv(in_channels=128, out_channels=1, kernel_size=5, stride=1, padding=2)
        )
        self.conv7 = nn.Sequential(
            DSConv(
                in_channels=128, out_channels=64, kernel_size=5, stride=4, padding=1
            ),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            DSConv(in_channels=64, out_channels=32, kernel_size=5, stride=4, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 1024), nn.Linear(1024, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        return mask, self.fc(x)
