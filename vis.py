from func import *
import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch.autograd import Variable
from models import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, default="./dataset/test_a/data/1_rain.png")
parser.add_argument(
    "--model_type",
    type=str,
    default="baseline",
    help="baseline or transformer or dsconv",
)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--model_path", type=str, default="./weights/baseline_gen.pkl")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = args.img_path
input = cv2.imread(img_path)
original = input.copy()
original = np.array(original)
input = cv2.resize(input, (224, 224))
input = np.array(input)
input_tensor = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0).float() / 255.0
if args.model_type == "baseline":
    model = Generator()
elif args.model_type == "transformer":
    model = GeneratorWithTransformer()
elif args.model_type == "dsconv":
    model = DSConvGenerator()
model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
model.eval()
model.to(device)

with torch.no_grad():
    if args.model_type != "transformer":
        mask, frame1, frame2, x = model(input_tensor.to(device))
        mask = mask[-1]
    else:
        mask, frame1, frame2, x = model(input_tensor.to(device))

get_heatmap(mask.squeeze().detach().cpu().numpy(), input, save_path=args.save_path)

clean_img_path = "./dataset/test_a/gt/1_clean.png"
clean_img = cv2.imread(clean_img_path)
clean_img = cv2.resize(clean_img, (224, 224))
clean_img = np.array(clean_img)

mask = get_mask(input, clean_img)
get_heatmap(mask, input, save_path="heatmap_gt.jpg")
