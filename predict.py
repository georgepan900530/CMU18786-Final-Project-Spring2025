# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

# Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
import os
import time

# Models lib
from models import *

# Metrics lib
from metrics import calc_psnr, calc_ssim, calc_lpips


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="baseline", help="baseline, dsconv, transformer"
    )
    parser.add_argument("--mode", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--ckpt_path", type=str, default="./weights/baseline_gen.pkl")
    
    # Transformer
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=1024,
        help="embedding dimension",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="number of heads",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="depth of transformer",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=4096,
        help="mlp dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout rate",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="patch size",
    )
    parser.add_argument(
        "--local_conv",
        action="store_true",
        help="use local conv for transformer",
    )
    args = parser.parse_args()
    return args


def align_to_four(img):
    # print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    # align to four
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    # print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(image, time_list):
    # img = np.array(image, dtype=np.float32)
    # img = img.transpose((2, 0, 1))
    # image = image[np.newaxis, :, :, :]
    # image = torch.from_numpy(image)
    # image = Variable(image).cuda()
    img = transforms.Resize((224, 224))(image)
    img = transforms.ToTensor()(img)
    img = img.cuda()
    img = img.unsqueeze(0)
    start_time = time.time()
    out = model(img)[-1]
    end_time = time.time()
    print(f"Time taken for prediction: {end_time - start_time} seconds")
    time_list.append(end_time - start_time)
    # out = out.cpu().data
    # out = out.numpy()
    out = out.detach().cpu().numpy()
    out = out.transpose((0, 2, 3, 1))
    out = out[0, :, :, :] * 255.0

    return out, time_list


if __name__ == "__main__":
    args = get_args()

    if args.model == "dsconv":
        model = DSConvGenerator().cuda()
    elif args.model == "transformer":
        model = GeneratorWithTransformer(local_conv=args.local_conv, num_heads=args.num_heads, embed_dim=args.embed_dim, depth=args.depth, mlp_dim=args.mlp_dim, dropout=args.dropout, patch_size=args.patch_size).cuda()
    else:
        model = Generator().cuda()
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    if args.mode == "demo":
        if not os.path.exists(args.output_dir):
            print("Output directory does not exist. Creating it now...")
            os.makedirs(args.output_dir)
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        time_list = []
        for i in range(num):
            print("Processing image: %s" % (input_list[i]))
            # img = cv2.imread(os.path.join(args.input_dir, input_list[i]))
            img = Image.open(os.path.join(args.input_dir, input_list[i])).convert("RGB")
            width, height = img.size
            # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            # img = align_to_four(img)
            result, _ = predict(img, time_list)
            result = result.astype(np.uint8)
            result = Image.fromarray(result)
            result = result.resize((width, height))
            img_name = input_list[i].split(".")[0]
            path = os.path.join(args.output_dir, img_name + ".jpg")
            # cv2.imwrite(path, result)
            result.save(path)

    elif args.mode == "test":
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        cumulative_lpips = 0
        time_list = []
        for i in range(num):
            print("Processing image: %s" % (input_list[i]))
            # img = cv2.imread(os.path.join(args.input_dir, input_list[i]))
            img = Image.open(os.path.join(args.input_dir, input_list[i])).convert("RGB")
            # gt = cv2.imread(os.path.join(args.gt_dir, gt_list[i]))
            gt = Image.open(os.path.join(args.gt_dir, gt_list[i])).convert("RGB")
            # img = align_to_four(img) # (480, 720)
            # gt = align_to_four(gt) # (480, 720)
            # The input to the model is (224, 224)
            # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            # gt = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_AREA)
            gt = transforms.Resize((224, 224))(gt)
            gt = np.array(gt, dtype=np.uint8)
            result, time_list = predict(img, time_list)
            result = np.array(result, dtype="uint8")
            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            cur_lpips = calc_lpips(gt, result)
            print(
                "PSNR is %.4f and SSIM is %.4f and LPIPS is %.4f"
                % (cur_psnr, cur_ssim, cur_lpips)
            )
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
            cumulative_lpips += cur_lpips
        print(
            "In testing dataset, PSNR is %.4f and SSIM is %.4f and LPIPS is %.4f"
            % (cumulative_psnr / num, cumulative_ssim / num, cumulative_lpips / num)
        )
        print(
            f"Average time taken for prediction: {sum(time_list) / len(time_list)} seconds"
        )

    else:
        print("Mode Invalid!")
