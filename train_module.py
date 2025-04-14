"""
This file is refactored from the unofficial implementation of the paper (https://github.com/shleecs/DeRaindrop_unofficial)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import cv2
from loss import *
from torchvision import transforms
import numpy as np
import cv2
import random
import time
import os
import argparse

from models import *
from func import *
from data.dataset_util import RainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter


class trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if opt.model_type == "baseline":
            self.net_G = DilatedGenerator().to(self.device)
            self.net_D = Discriminator().to(self.device)
        elif opt.model_type == "dsconv":
            self.net_G = DSConvGenerator().to(self.device)
            self.net_D = DSConvDiscriminator().to(self.device)
        elif opt.model_type == "transformer":
            self.net_G = GeneratorWithTransformer(
                embed_dim=opt.embed_dim,
                num_heads=opt.num_heads,
                depth=opt.depth,
                mlp_dim=opt.mlp_dim,
                dropout=opt.dropout,
                patch_size=opt.patch_size,
                local_conv=opt.local_conv,
            ).to(self.device)
            self.net_D = Discriminator().to(self.device)
        print(f"Model type: {opt.model_type}")
        if opt.load != -1:
            G_ckpt = os.path.join(opt.checkpoint_dir, f"G_epoch_{opt.load}.pth")
            D_ckpt = os.path.join(opt.checkpoint_dir, f"D_epoch_{opt.load}.pth")
            self.net_G.load_state_dict(torch.load(G_ckpt))
            self.net_D.load_state_dict(torch.load(D_ckpt))
            print("Successfully load the model")
        self.optim1 = torch.optim.Adam(
            # filter(lambda p: p.requires_grad, self.net_G.parameters()),
            self.net_G.parameters(),
            lr=opt.lr,
            betas=(0.5, 0.99),
        )
        self.optim2 = torch.optim.Adam(
            # filter(lambda p: p.requires_grad, self.net_D.parameters()),
            self.net_D.parameters(),
            lr=opt.lr,
            betas=(0.5, 0.99),
        )
        self.start = opt.load
        self.iter = opt.iter
        self.batch_size = opt.batch_size
        self.early_stop = opt.early_stop
        # train_dataset = RainDataset("./dataset", is_eval=False)
        # valid_dataset = RainDataset("./dataset", is_eval=True)
        if opt.aug:
            print("Applying data augmentation")
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    # Apply non-flipping transforms so that we dont need to do the same for the ground truth
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                ]
            )
        else:
            transform = None
        train_dataset = RainDataset(
            opt, is_eval=False, transform=transform, horizontal_flip=True
        )
        valid_dataset = RainDataset(
            opt, is_eval=True, transform=transform, horizontal_flip=False
        )
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=opt.batch_size, num_workers=2, pin_memory=True
        )

        print("# train set : {}".format(train_size))
        print("# eval set : {}".format(valid_size))

        self.expr_dir = opt.checkpoint_dir

        # Attention Loss
        if opt.model_type != "transformer":
            self.criterionAtt = AttentionLoss(theta=0.8, iteration=4)
        else:
            self.criterionAtt = AttentionLossWithTransformer()
        # GAN Loss
        if opt.gan_loss == "bce":
            self.criterionGAN = GANLoss(real_label=1.0, fake_label=0.0)
        elif opt.gan_loss == "mse":
            self.criterionGAN = AdvancedGANLoss(mode="mse")
        elif opt.gan_loss == "hinge":
            self.criterionGAN = AdvancedGANLoss(mode="hinge")
        elif opt.gan_loss == "wasserstein":
            self.criterionGAN = AdvancedGANLoss(mode="wasserstein")
            self.lambda_gp = 10.0
        else:
            raise ValueError(f"Invalid GAN loss type: {opt.gan_loss}")
        # Perceptual Loss
        self.criterionPL = PerceptualLoss()
        # Multiscale Loss
        self.criterionML = MultiscaleLoss(ld=[0.6, 0.8, 1.0])
        # MAP Loss
        self.criterionMAP = MAPLoss(gamma=0.05)
        # MSE Loss
        self.criterionMSE = nn.MSELoss().to(self.device)

        if opt.scheduler:
            print("Using cosine annealing scheduler")
            self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim1, T_max=self.iter, verbose=True
            )
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim2, T_max=self.iter, verbose=True
        )
        self.out_path = opt.checkpoint_dir

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN-GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(self.device)

        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)

        # Get discriminator output for interpolated images
        d_map_interpolates, d_interpolates = self.net_D(interpolates)

        # Get gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten gradients to calculate their norm
        gradients = gradients.contiguous().view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def forward_process(self, I_, GT, is_train=True):
        # I_: input raindrop image
        # A_: attention map(Mask_list) from ConvLSTM
        # M_: mask GT
        # O_: output image of the autoencoder
        # T_: GT

        # M_ = []
        # for i in range(I_.shape[0]):
        #     I_img = I_[i].permute(1, 2, 0).cpu().numpy()
        #     GT_img = GT[i].permute(1, 2, 0).cpu().numpy()
        #     M_.append(get_mask(I_img, GT_img))

        # 将 mask 从 [B, H, W, 1] 转换为 [B, 1, H, W]
        # M_ = torch.from_numpy(np.array(M_)).permute(0, 3, 1, 2).float().to(self.device)
        # I_ = I_.to(self.device)
        # GT_ = GT.to(self.device)

        M_ = []
        for i in range(I_.shape[0]):
            M_.append(get_mask(np.array(I_[i]), np.array(GT[i])))
        M_ = np.array(M_)
        M_ = torch_variable(M_, is_train)
        # I_ = torch_variable(I_, is_train)
        # GT_ = torch_variable(GT, is_train)
        GT_ = Variable(GT, requires_grad=is_train).to(self.device)
        I_ = Variable(I_, requires_grad=is_train).to(self.device)
        A_, t1, t2, t3 = self.net_G(I_)
        # print 'mask len', len(A_)
        S_ = [t1, t2, t3]
        O_ = t3

        loss = self.criterionMSE(O_, GT_.detach())

        if is_train:
            # attention_loss
            loss_att = self.criterionAtt(A_, M_.detach())

            # perceptual_loss O_: generation, T_: GT
            loss_PL = self.criterionPL(O_, GT_.detach())

            # Multiscale_loss
            loss_ML = self.criterionML(S_, GT)

            # print('t3', t3.shape)
            # D(Fake)

            D_map_O, D_fake = self.net_D(t3.detach())
            # D(Real)
            # GT = torch_variable(GT,is_train, is_grad=True)
            D_map_R, D_real = self.net_D(GT_)
            if self.opt.model_type != "transformer":
                loss_MAP = self.criterionMAP(D_map_O, D_map_R, A_[-1].detach())
            else:
                loss_MAP = self.criterionMAP(D_map_O, D_map_R, A_.detach())

            # loss_fake = self.criterionGAN(
            #     D_fake, is_real=False
            # )  # BCE 1, D_fake -(log(1-fake))
            # loss_real = self.criterionGAN(
            #     D_real, is_real=True
            # )  # BCE 0, D_real -log(real)
            # # D_real, 1
            # loss_D = loss_real + loss_fake + loss_MAP
            # # print (loss_gen_D), (loss_att), (loss_ML), (loss_PL)
            # loss_G = 0.01 * (-loss_fake) + loss_att + loss_ML + loss_PL
            # Calculate losses based on the GAN type
            if self.opt.gan_loss == "wasserstein":
                # Wasserstein loss with gradient penalty
                loss_fake = self.criterionGAN(D_fake, is_real=False)
                loss_real = self.criterionGAN(D_real, is_real=True)

                # Calculate gradient penalty
                gradient_penalty = self.compute_gradient_penalty(GT_, t3.detach())

                # Discriminator loss
                loss_D = (
                    loss_real + loss_fake + self.lambda_gp * gradient_penalty + loss_MAP
                )

                # Generator loss (note: for WGAN we want to maximize D(G(z))
                loss_G = 0.01 * (-loss_fake) + loss_att + loss_ML + loss_PL
            else:
                # Standard GAN losses
                loss_fake = self.criterionGAN(D_fake, is_real=False)
                loss_real = self.criterionGAN(D_real, is_real=True)
                loss_D = loss_real + loss_fake + loss_MAP
                loss_G = 0.01 * (-loss_fake) + loss_att + loss_ML + loss_PL

            output = [loss_G, loss_D, loss_PL, loss_ML, loss_att, loss_MAP, loss]
        else:
            output = loss
        return output

    def train_start(self):
        loss_sum = 0.0
        valid_loss_sum = 0.0
        # I_: input raindrop image
        # A_: attention map(Mask_list) from ConvLSTM
        # M_: mask GT
        # O_: output image of the autoencoder
        # T_: GT
        writer = SummaryWriter()
        count = 0
        before_loss = 10000000
        early_stop = 0
        for epoch in range(self.start, self.iter + 1):
            self.net_G.train()
            self.net_D.train()
            for i, data in enumerate(self.train_loader):
                count += 1
                I_, GT_ = data
                # print 'GT:',GT_.shape
                loss_G, loss_D, loss_PL, loss_ML, loss_att, loss_MAP, MSE_loss = (
                    self.forward_process(I_, GT_)
                )
                # print loss_G

                self.optim1.zero_grad()
                loss_G.backward(retain_graph=True)
                self.optim1.step()

                self.optim2.zero_grad()
                loss_D.backward()
                self.optim2.step()

                if count % 20 == 0:
                    print(
                        "epoch: "
                        + str(epoch)
                        + " count: "
                        + str(count)
                        + " loss G: {:.4f}".format(loss_G.item())
                        + " loss_D: {:.4f}".format(loss_D.item())
                        + " loss_MSE: {:.4f}".format(MSE_loss.item())
                    )
                    print(
                        "loss_PL:{:.4f}".format(loss_PL.item())
                        + " loss_ML:{:.4f}".format(loss_ML.item())
                        + " loss_Att:{:.4f}".format(loss_att.item())
                        + " loss_MAP:{:.4f}".format(loss_MAP.item())
                    )
                    writer.add_scalar("loss_G", loss_G.item(), count)
                    writer.add_scalar("loss_D", loss_D.item(), count)

            step = 0
            self.net_G.eval()
            self.net_D.eval()
            with torch.no_grad():
                for i, data in enumerate(self.valid_loader):
                    I_, GT_ = data
                    if i == 0:
                        valid_loss_sum = self.forward_process(I_, GT_, is_train=False)
                    else:
                        valid_loss_sum += self.forward_process(I_, GT_, is_train=False)
                    step += 1

            avg_valid = valid_loss_sum.item() / step
            print("epoch_" + str(epoch) + " valid_loss: {:.4f}".format(avg_valid))
            writer.add_scalar("validation_loss", avg_valid, epoch)
            valid_loss_sum = valid_loss_sum.item() / step
            if before_loss / valid_loss_sum > 1.01:
                early_stop = 0
                before_loss = valid_loss_sum
                print("save model " + "!" * 10)
                if not os.path.exists(self.out_path):
                    os.system("mkdir -p {}".format(self.out_path))
                w_name = "G_epoch_{}.pth".format(epoch)
                save_path = os.path.join(self.out_path, w_name)
                torch.save(self.net_G.state_dict(), save_path)
                w_name = "D_epoch_{}.pth".format(epoch)
                save_path = os.path.join(self.out_path, w_name)
                torch.save(self.net_D.state_dict(), save_path)
            else:
                early_stop += 1
                if early_stop > self.early_stop:
                    print(
                        f"early stop after {self.early_stop} epochs without improvement :("
                    )
                    break
            valid_loss_sum = 0.0
            if self.opt.scheduler:
                self.scheduler_G.step()
                self.scheduler_D.step()
        torch.save(self.net_G.state_dict(), os.path.join(self.out_path, "G_final.pth"))
        torch.save(self.net_D.state_dict(), os.path.join(self.out_path, "D_final.pth"))
        writer.export_scalars_to_json("./attention_video_restoration.json")
        writer.close()
        return
