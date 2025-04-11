# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# Tools lib
import numpy as np
import cv2
import random
import time
import os

# Custom modules
try:
    from .modules import DSConv, RainDropMaskDecoder
except ImportError:
    from modules import DSConv, RainDropMaskDecoder

# Set iteration time
ITERATION = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU())
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.conv_i = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 5, 1, 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4), nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8), nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16), nn.ReLU()
        )
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.outframe1 = nn.Sequential(nn.Conv2d(256, 3, 3, 1, 1), nn.ReLU())
        self.outframe2 = nn.Sequential(nn.Conv2d(128, 3, 3, 1, 1), nn.ReLU())
        self.output = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).to(device) / 2.0
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        # The author use generative model to first predict N masks for raindrop
        mask_list = []
        for i in range(ITERATION):
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        # The last mask is concatenated with the input and passed to the autoencoder
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return mask_list, frame1, frame2, x


class DSConvGenerator(nn.Module):
    def __init__(self):
        super(DSConvGenerator, self).__init__()
        self.det_conv0 = nn.Sequential(DSConv(4, 32, 3, 1, 1), nn.ReLU())
        self.det_conv1 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )

        self.conv_i = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Conv2d(32, 1, 3, 1, 1)

        self.conv1 = nn.Sequential(DSConv(4, 64, 5, 1, 2), nn.ReLU())
        self.conv2 = nn.Sequential(DSConv(64, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(DSConv(128, 128, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(DSConv(128, 256, 3, 2, 1), nn.ReLU())
        self.conv5 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())
        self.conv6 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())

        # self.diconv1 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.ReLU()
        # )
        # self.diconv2 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 4, dilation=4), nn.ReLU()
        # )
        # self.diconv3 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 8, dilation=8), nn.ReLU()
        # )
        # self.diconv4 = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 16, dilation=16), nn.ReLU()
        # )
        self.diconv1 = nn.Sequential(DSConv(256, 256, 3, 1, 2, dilation=2), nn.ReLU())
        self.diconv2 = nn.Sequential(DSConv(256, 256, 3, 1, 4, dilation=4), nn.ReLU())
        self.diconv3 = nn.Sequential(DSConv(256, 256, 3, 1, 8, dilation=8), nn.ReLU())
        self.diconv4 = nn.Sequential(DSConv(256, 256, 3, 1, 16, dilation=16), nn.ReLU())

        self.conv7 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())
        self.conv8 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(DSConv(128, 128, 3, 1, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(DSConv(64, 32, 3, 1, 1), nn.ReLU())
        self.outframe1 = nn.Sequential(DSConv(256, 3, 3, 1, 1), nn.ReLU())
        self.outframe2 = nn.Sequential(DSConv(128, 3, 3, 1, 1), nn.ReLU())
        # 最终输出层，将通道数从 64 转换为 3
        self.output = nn.Sequential(DSConv(32, 3, 3, 1, 1))

    def forward(self, input):
        device = input.device
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col, device=device) / 2.0)
        h = Variable(torch.zeros(batch_size, 32, row, col, device=device))
        c = Variable(torch.zeros(batch_size, 32, row, col, device=device))
        mask_list = []

        for iter_idx in range(ITERATION):
            # print(f"🔁 ITERATION {iter_idx}")
            # print("   input shape:", input.shape)
            # print("   mask  shape:", mask.shape)
            # try:
            #     cat_x = torch.cat((input, mask), 1)
            #     print("   cat   shape:", cat_x.shape)
            # except Exception as e:
            #     print("  cat failed:", str(e))
            #     raise
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)
            i_gate = self.conv_i(x)
            f_gate = self.conv_f(x)
            g_gate = self.conv_g(x)
            o_gate = self.conv_o(x)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)

        mask = mask_list[-1]

        # try:
        #     cat_x = torch.cat((input, mask), 1)
        #     print("cat  :", cat_x.shape)
        # except Exception as e:
        #     print("❌ concat failed:", str(e))
        #     raise
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        # x = self.deconv2(x)
        # x = x + res1

        # x = x.clone()
        # x = self.out_final(x)
        return mask_list, frame1, frame2, x

    # class DSConvGenerator(nn.Module):
    def __init__(self):
        super(DSConvGenerator, self).__init__()
        self.det_conv0 = nn.Sequential(DSConv(4, 32, 3, 1, 1), nn.ReLU())
        self.det_conv1 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            DSConv(32, 32, 3, 1, 1), nn.ReLU(), DSConv(32, 32, 3, 1, 1), nn.ReLU()
        )

        self.conv_i = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Conv2d(32, 1, 3, 1, 1)

        self.conv1 = nn.Sequential(DSConv(4, 64, 5, 1, 2), nn.ReLU())
        self.conv2 = nn.Sequential(DSConv(64, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(DSConv(128, 128, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(DSConv(128, 256, 3, 2, 1), nn.ReLU())
        self.conv5 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())
        self.conv6 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())

        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4), nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8), nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16), nn.ReLU()
        )

        self.conv7 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())
        self.conv8 = nn.Sequential(DSConv(256, 256, 3, 1, 1), nn.ReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )

        self.conv9 = nn.Sequential(DSConv(128, 64, 3, 1, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.outframe1 = nn.Conv2d(256, 3, 3, 1, 1)
        self.outframe2 = nn.Conv2d(64, 3, 3, 1, 1)
        # 最终输出层，将通道数从 64 转换为 3
        self.out_final = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, input):
        device = input.device
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = torch.ones(batch_size, 1, row, col, device=device) / 2.0
        h = torch.zeros(batch_size, 32, row, col, device=device)
        c = torch.zeros(batch_size, 32, row, col, device=device)
        mask_list = []

        for iter_idx in range(ITERATION):
            # print(f"🔁 ITERATION {iter_idx}")
            # print("   input shape:", input.shape)
            # print("   mask  shape:", mask.shape)
            # try:
            #     cat_x = torch.cat((input, mask), 1)
            #     print("   cat   shape:", cat_x.shape)
            # except Exception as e:
            #     print("  cat failed:", str(e))
            #     raise
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x.clone()
            x = F.relu(self.det_conv1(x) + resx)
            resx = x.clone()
            x = F.relu(self.det_conv2(x) + resx)
            resx = x.clone()
            x = F.relu(self.det_conv3(x) + resx)
            resx = x.clone()
            x = F.relu(self.det_conv4(x) + resx)
            resx = x.clone()
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)
            i_gate = self.conv_i(x)
            f_gate = self.conv_f(x)
            g_gate = self.conv_g(x)
            o_gate = self.conv_o(x)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * torch.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)

        mask = mask_list[-1]

        # try:
        #     cat_x = torch.cat((input, mask), 1)
        #     print("cat  :", cat_x.shape)
        # except Exception as e:
        #     print("❌ concat failed:", str(e))
        #     raise
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x.clone()
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x.clone()
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1

        x = x.clone()
        x = self.out_final(x)
        return mask_list, frame1, frame2, x


class GeneratorWithTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        num_heads=8,
        depth=12,
        mlp_dim=4096,
        dropout=0.0,
        patch_size=16,
        local_conv=False,
    ):
        super(GeneratorWithTransformer, self).__init__()
        self.det_conv0 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU())
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.conv_i = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 5, 1, 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4), nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8), nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16), nn.ReLU()
        )
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.outframe1 = nn.Sequential(nn.Conv2d(256, 3, 3, 1, 1), nn.ReLU())
        self.outframe2 = nn.Sequential(nn.Conv2d(128, 3, 3, 1, 1), nn.ReLU())
        self.output = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1))
        self.raindrop_decoder = RainDropMaskDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_dim=mlp_dim,
            dropout=dropout,
            patch_size=patch_size,
            local_conv=local_conv,
        )
        for name, param in self.raindrop_decoder.params.items():
            print(f"{name}: {param}")

    def forward(self, input):
        mask = self.raindrop_decoder(input)
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return mask, frame1, frame2, x


class DilatedGenerator(nn.Module):
    def __init__(self):
        super(DilatedGenerator, self).__init__()
        self.det_conv0 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU())
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU()
        )
        self.conv_i = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 5, 1, 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation=2), nn.ReLU()
        )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation=4), nn.ReLU()
        )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation=8), nn.ReLU()
        )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation=16), nn.ReLU()
        )
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU(),
        )
        self.conv10 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.outframe1 = nn.Sequential(nn.Conv2d(256, 3, 3, 1, 1), nn.ReLU())
        self.outframe2 = nn.Sequential(nn.Conv2d(128, 3, 3, 1, 1), nn.ReLU())
        self.output = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).to(device) / 2.0
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        # The author use generative model to first predict N masks for raindrop
        mask_list = []
        for i in range(ITERATION):
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        # The last mask is concatenated with the input and passed to the autoencoder
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return mask_list, frame1, frame2, x


if __name__ == "__main__":
    # dsconv_generator = DSConvGenerator()
    # net = Generator()
    # input = torch.randn(1, 3, 224, 224)
    # mask_list, frame1, frame2, x = net(input)
    # print(mask_list[0].shape, frame1.shape, frame2.shape, x.shape)
    # mask_list, frame1, frame2, x = dsconv_generator(input)
    # print(mask_list[0].shape, frame1.shape, frame2.shape, x.shape)
    generator = DilatedGenerator()
    img_path = "./dataset/train/data/1_rain.png"
    input = cv2.imread(img_path)
    input = cv2.resize(input, (224, 224))
    input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask, frame1, frame2, x = generator(input)
    print(mask[0].shape, frame1.shape, frame2.shape, x.shape)
