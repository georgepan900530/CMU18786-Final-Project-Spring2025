import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RainDataset(Dataset):
    def __init__(self, root_dir, is_eval=False, transform=None):
        self.root_dir = os.path.join(root_dir, "test_a" if is_eval else "train")
        self.data_dir = os.path.join(self.root_dir, "data")
        self.gt_dir = os.path.join(self.root_dir, "gt")
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        self.filenames = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data_path = os.path.join(self.data_dir, filename)

        gt_filename = filename.replace("_rain", "_clean")
        gt_path = os.path.join(self.gt_dir, gt_filename)

        data_img = Image.open(data_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        data_tensor = self.transform(data_img)
        gt_tensor = self.transform(gt_img)

        # print("data tensor shape:", data_tensor.shape)

        return data_tensor, gt_tensor
