import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import glob

# class RainDataset(Dataset):
#     def __init__(self, root_dir, is_eval=False, transform=None):
#         self.root_dir = os.path.join(root_dir, "test_a" if is_eval else "train")
#         self.data_dir = os.path.join(self.root_dir, "data")
#         self.gt_dir = os.path.join(self.root_dir, "gt")
#         if transform:
#             self.transform = transform
#         else:
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize((224 , 224)),
#                     transforms.ToTensor(),
#                 ]
#             )

#         self.filenames = sorted(os.listdir(self.data_dir))

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         filename = self.filenames[idx]
#         data_path = os.path.join(self.data_dir, filename)

#         gt_filename = filename.replace("_rain", "_clean")
#         gt_path = os.path.join(self.gt_dir, gt_filename)

#         data_img = Image.open(data_path).convert("RGB")
#         gt_img = Image.open(gt_path).convert("RGB")

#         data_tensor = self.transform(data_img)
#         gt_tensor = self.transform(gt_img)

#         # print("data tensor shape:", data_tensor.shape)

#         return data_tensor, gt_tensor

class RainDataset(Dataset):
    def __init__(self, opt, is_eval=False, is_test=False):
        super(RainDataset, self).__init__()

        if is_test:
            self.dataset = opt.test_dataset
        elif is_eval:
            self.dataset = opt.eval_dataset
        else:
            self.dataset = opt.train_dataset
        # dataset = open(self.dataset, 'r').read().split()
        self.img_list = sorted(glob.glob(self.dataset+'/data/*'))
        self.gt_list = sorted(glob.glob(self.dataset+'/gt/*'))
   
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        gt_name = self.gt_list[idx]

        img = cv2.imread(img_name,-1)
        gt = cv2.imread(gt_name,-1)

        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (224,224), interpolation=cv2.INTER_AREA)
        # img = np.asarray(img).transpose((2,0,1))
        # gt = np.asarray(gt).transpose((2,0,1))

        if img.dtype == np.uint8:
            img = (img / 255.0).astype('float32')
        if gt.dtype == np.uint8:
            gt = (gt / 255.0).astype('float32')

        return [img,gt]
