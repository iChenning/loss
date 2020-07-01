from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        assert os.path.exists(txt_path), "不存在"+txt_path
        f = open(txt_path, 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return (img,label)

    def __len__(self):
        return len(self.imgs)


class MyDatasetOfBYOL(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        assert os.path.exists(txt_path), "不存在"+txt_path
        f = open(txt_path, 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        assert self.transform, "byol need transform"
        img1 = self.transform(img)
        img2 = self.transform(img)

        return (img1, img2, label)

    def __len__(self):
        return len(self.imgs)
