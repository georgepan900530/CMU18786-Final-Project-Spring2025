"""
This file is refactored from the unofficial implementation of the paper (https://github.com/shleecs/DeRaindrop_unofficial)
"""

import torch
import torch.nn as nn
import torchvision
import cv2
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
import numpy as np
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        # self.loss = nn.MSELoss().cuda()
        self.loss = nn.BCELoss().cuda()

    def convert_tensor(self, input, is_real):
        device = input.device
        if is_real:
            return Variable(torch.FloatTensor(input.size()).fill_(self.real_label)).to(
                device
            )
        else:
            return Variable(torch.FloatTensor(input.size()).fill_(self.fake_label)).to(
                device
            )

    def __call__(self, input, is_real):
        return self.loss(input, self.convert_tensor(input, is_real))
    
class AdvancedGANLoss(nn.Module):
    """Advanced GAN loss implementation with label smoothing, noise injection, and multiple loss functions.
    
    Features:
    - Label smoothing for more stable training
    - Optional noise injection to prevent discriminator overconfidence
    - Support for multiple loss functions (BCE, MSE, hinge, wasserstein)
    - Device-agnostic implementation
    """
    def __init__(self, mode='bce', label_smoothing=0.1, noise_std=0.05, use_noise=True):
        """
        Args:
            mode (str): Loss function to use - 'bce', 'mse', 'hinge', or 'wasserstein'
            label_smoothing (float): Amount of label smoothing (0.0 to 0.5)
            noise_std (float): Standard deviation of noise added to labels
            use_noise (bool): Whether to use noise injection
        """
        super(GANLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode.lower()
        self.label_smoothing = label_smoothing
        self.noise_std = noise_std
        self.use_noise = use_noise
        
        # Real and fake labels with smoothing
        self.real_label = 1.0 - label_smoothing
        self.fake_label = 0.0 + label_smoothing
        
        # Register buffers for the label tensors
        self.register_buffer('real_tensor', torch.FloatTensor(1).fill_(self.real_label))
        self.register_buffer('fake_tensor', torch.FloatTensor(1).fill_(self.fake_label))
        
        # Set up the appropriate loss function
        if self.mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss() if mode == 'bce_logits' else nn.BCELoss()
        elif self.mode == 'mse':
            self.loss = nn.MSELoss()
        elif self.mode in ['hinge', 'wasserstein']:
            # For hinge and wasserstein, we'll implement custom loss calculations
            self.loss = None
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")
            
    def get_target_tensor(self, input, is_real):
        """Generate target tensors with optional noise injection."""
        if is_real:
            target = self.real_tensor.expand_as(input)
        else:
            target = self.fake_tensor.expand_as(input)
            
        # Add noise to labels if enabled
        if self.use_noise and self.training:
            noise = torch.randn_like(target) * self.noise_std
            # Clamp to ensure we stay in valid probability range for BCE
            if self.mode == 'bce':
                target = torch.clamp(target + noise, min=0.0, max=1.0)
            else:
                target = target + noise
                
        return target
    
    def __call__(self, input, is_real):
        """Calculate loss based on discriminator predictions and whether they should be real/fake.
        
        Args:
            input: Output from the discriminator
            is_real: Whether the input should be classified as real (True) or fake (False)
        """
        # For wasserstein loss
        if self.mode == 'wasserstein':
            return -torch.mean(input) if is_real else torch.mean(input)
            
        # For hinge loss
        elif self.mode == 'hinge':
            if is_real:
                return torch.mean(F.relu(1.0 - input))
            else:
                return torch.mean(F.relu(1.0 + input))
                
        # For BCE and MSE losses
        else:
            target_tensor = self.get_target_tensor(input, is_real)
            return self.loss(input, target_tensor)
            
    def train(self, mode=True):
        """Override train method to ensure proper behavior during eval."""
        super(GANLoss, self).train(mode)
        return self


class AttentionLoss(nn.Module):
    def __init__(self, theta=0.8, iteration=4):
        super(AttentionLoss, self).__init__()
        self.theta = theta
        self.iteration = iteration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss().to(self.device)

    def __call__(self, A_, M_):
        loss_ATT = None
        for i in range(1, self.iteration + 1):
            if i == 1:
                loss_ATT = pow(self.theta, float(self.iteration - i)) * self.loss(
                    A_[i - 1], M_
                )
            else:
                loss_ATT += pow(self.theta, float(self.iteration - i)) * self.loss(
                    A_[i - 1], M_
                )
        return loss_ATT
    
class AttentionLossWithTransformer(nn.Module):
    def __init__(self):
        super(AttentionLossWithTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss().to(self.device)
        
    def __call__(self, A_, M_):
        return self.loss(A_, M_)


# VGG16 pretrained on Imagenet
def trainable(net, trainable):
    for param in net.parameters():
        param.requires_grad = trainable


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = vgg16(pretrained=True).to(device)
        trainable(self.model, False)

        self.loss = nn.MSELoss().cuda()
        self.vgg_layers = self.model.features
        self.layer_name_mapping = {
            "1": "relu1_1",
            "3": "relu1_2",
            "6": "relu2_1",
            "8": "relu2_2",
        }

    def get_layer_output(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

    def __call__(self, O_, T_):
        o = self.get_layer_output(O_)
        t = self.get_layer_output(T_)
        loss_PL = None
        for i in range(len(t)):
            if i == 0:
                loss_PL = self.loss(o[i], t[i]) / float(len(t))
            else:
                loss_PL += self.loss(o[i], t[i]) / float(len(t))
        return loss_PL


class MultiscaleLoss(nn.Module):
    def __init__(self, ld=[0.6, 0.8, 1.0], batch=1):
        super(MultiscaleLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss()  # 不需要额外调用 .cuda()，后面会用 .to(self.device)
        self.loss = nn.MSELoss().to(self.device)
        self.ld = ld
        self.batch = batch

    def forward(self, S_, gt):
        """
        S_ : list of generator outputs at different scales, e.g.,
             S_[0]: [B, 3, H_small, W_small]
             S_[1]: [B, 3, H_mid,   W_mid]
             S_[2]: [B, 3, H_orig,  W_orig]
        gt : ground truth tensor, shape [B, 3, H, W]
        """
        # 定义各尺度的缩放比例（和生成器对应）
        scales = [0.25, 0.5, 1.0]
        T_ = []
        for scale in scales:
            T_.append(
                F.interpolate(
                    gt, scale_factor=scale, mode="bilinear", align_corners=False
                )
            )

        loss_ML = 0.0
        for i in range(len(self.ld)):
            loss_ML += self.ld[i] * self.loss(S_[i], T_[i].to(self.device))
        return loss_ML

# class MultiscaleLoss(nn.Module):
#     def __init__(self, ld=[0.6,0.8,1.0],batch=1):
#         super(MultiscaleLoss, self).__init__()
#         self.loss = nn.MSELoss().cuda()
#         self.ld = ld
#         self.batch=batch
#     def __call__(self, S_, gt):
#         #1,128,256,3
#         T_ = []
#         # print S_[0].shape[0]
#         for i in range(S_[0].shape[0]):
#             temp = []
#             x = (np.array(gt[i])*255.).astype(np.uint8)
#             # print (x.shape, x.dtype)
#             t = cv2.resize(x, None, fx=1.0/4.0,fy=1.0/4.0, interpolation=cv2.INTER_AREA)
#             t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
#             temp.append(t)
#             t = cv2.resize(x, None, fx=1.0/2.0,fy=1.0/2.0, interpolation=cv2.INTER_AREA)
#             t = np.expand_dims((t/255.).astype(np.float32).transpose(2,0,1),axis=0)
#             temp.append(t)
#             x = np.expand_dims((x/255.).astype(np.float32).transpose(2,0,1),axis=0)
#             temp.append(x)
#             T_.append(temp)
#         temp_T = []
#         for i in range(len(self.ld)):
#             # if self.batch == 1:
#             #     temp_T.append(Variable(torch.from_numpy(T_[0][i])).cuda())
#             # else:
#             for j in range((S_[0].shape[0])):
#                 if j == 0:
#                     x = T_[j][i]
#                 else:
#                     x = np.concatenate((x, T_[j][i]), axis=0)
#             temp_T.append(Variable(torch.from_numpy(x)).cuda())
#         T_ = temp_T
#         loss_ML = None
#         for i in range(len(self.ld)):
#             if i == 0: 
#                 loss_ML = self.ld[i] * self.loss(S_[i], T_[i])
#             else:
#                 loss_ML += self.ld[i] * self.loss(S_[i], T_[i])
        
#         return loss_ML/float(S_[0].shape[0])


class MAPLoss(nn.Module):
    def __init__(self, gamma=0.05):
        super(MAPLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss().to(device)
        self.gamma = gamma

    # D_map_O, D_map_R
    def __call__(self, D_O, D_R, A_N):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Z = Variable(torch.zeros(D_R.shape)).to(device)
        D_A = self.loss(D_O, A_N)
        D_Z = self.loss(D_R, Z)
        return self.gamma * (D_A + D_Z)
