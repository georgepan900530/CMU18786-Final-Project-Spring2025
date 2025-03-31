import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RainDataset(Dataset):
    def __init__(self, root_dir, is_eval=False, transform=None):
        self.root_dir = os.path.join(root_dir, "test_a" if is_eval else "train")
        self.data_dir = os.path.join(self.root_dir, "data")
        self.gt_dir = os.path.join(self.root_dir, "gt")
        self.transform = transform or transforms.ToTensor()

        self.filenames = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data_path = os.path.join(self.data_dir, filename)

        gt_filename = filename.replace("_rain", "_clean")
        gt_path = os.path.join(self.gt_dir, gt_filename)

        data_img = Image.open(data_path).convert("RGB").resize((480, 320))
        gt_img = Image.open(gt_path).convert("RGB").resize((480, 320))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        data_tensor = transform(data_img)
        gt_tensor = transform(gt_img)

        # print("data tensor shape:", data_tensor.shape)

        return data_tensor, gt_tensor
