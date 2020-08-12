from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import os
from PIL import Image
from utils.transforms import transform
import random


def dataloader(opt):
    train_trans, test_trans = transform()
    read_train = opt.read_data.train
    if opt.read_data.is_disk:
        trainset = MyDataset(txt_path=read_train.file_path, transform=train_trans)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./Data', train=True, download=False, transform=train_trans)
    trainloader = DataLoader(trainset, batch_size=read_train.batch_size, shuffle=read_train.shuffle)

    read_test = opt.read_data.test
    if opt.read_data.is_disk:
        testset = MyDataset(txt_path=read_test.file_path, transform=test_trans)
    else:
        testset = torchvision.datasets.CIFAR10(root='./Data', train=False, download=False, transform=test_trans)
    testloader = DataLoader(testset, batch_size=read_test.batch_size, shuffle=read_test.shuffle)

    return (trainloader, testloader)


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        assert os.path.exists(txt_path), "不存在" + txt_path
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

        return (img, label)

    def __len__(self):
        return len(self.imgs)


class super_BYOL(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        assert os.path.exists(txt_path), "不存在" + txt_path
        f = open(txt_path, 'r')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        imgs_all = []
        for index1, img1 in enumerate(imgs):
            f_path1, label1 = img1
            for index2 in range(index1, len(imgs)):
                f_path2, label2 = imgs[index2]
                if label2 == label1 and index2 != index1:
                    imgs_all.append((f_path1, label1, f_path2, label2))

        self.imgs = imgs_all
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        f_path1, label1, f_path2, label2 = self.imgs[index]
        img1 = Image.open(f_path1).convert("RGB")
        img2 = Image.open(f_path2).convert("RGB")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2)

    def __len__(self):
        return len(self.imgs)


class unsup_BYOL(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        assert os.path.exists(txt_path), "不存在" + txt_path
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
        f_path1, label1 = self.imgs[index]
        img = Image.open(f_path1).convert("RGB")
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return (img1, img2)

    def __len__(self):
        return len(self.imgs)