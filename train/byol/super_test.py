from utils.my_dataset import MyDataset
from torch.utils.data import DataLoader
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from tensorboardX import SummaryWriter
import os
import argparse
from importlib.machinery import SourceFileLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(nn.Module):
    def __init__(self, opt, dim_mlp=128, m=0.996):
        super(BYOL, self).__init__()
        self.m = m

        from moduls.modul_net5 import Net5
        from moduls.modul_SideNet5 import SideNet5
        from moduls.modul_resnet22 import ResNet22
        from moduls.modul_SideRes22 import SideResNet22
        from moduls.modul_resnet26 import ResNet26
        from moduls.modul_ACRes26 import ACRes26

        if opt.train.feature_net == 'Net5':
            self.online_f = Net5()
            self.target_f = Net5()
        elif opt.train.feature_net == 'Net5_Side':
            self.online_f = SideNet5(opt)
            self.target_f = SideNet5(opt)
        elif opt.train.feature_net == 'Resnet22':
            self.online_f = ResNet22()
            self.target_f = ResNet22()
        elif opt.train.feature_net == 'Res22_Side':
            self.online_f = SideResNet22(opt)
            self.target_f = SideResNet22(opt)
        elif opt.train.feature_net == 'Resnet26':
            self.online_f = ResNet26()
            self.target_f = ResNet26()
        else:
            self.online_f = ACRes26()
            self.target_f = ACRes26()
        print("online_f is target_f?:", self.online_f is self.target_f)
        for param_online_f, param_target_f in zip(self.online_f.parameters(), self.target_f.parameters()):
            param_target_f.data.copy_(param_online_f.data)  # initialize
            param_target_f.requires_grad = False  # not update by gradient

        self.online_g = nn.Sequential(nn.Linear(dim_mlp, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, dim_mlp))
        self.target_g = nn.Sequential(nn.Linear(dim_mlp, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, dim_mlp))
        print("online_g is target_g?:", self.online_g is self.target_g)
        for param_online_g, param_target_g in zip(self.online_g.parameters(), self.target_g.parameters()):
            param_target_g.data.copy_(param_online_g.data)  # initialize
            param_target_g.requires_grad = False  # not update by gradient

        self.online_q = nn.Sequential(nn.Linear(dim_mlp, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, dim_mlp))

    @torch.no_grad()
    def _momentum_update_target(self):
        for param_online_f, param_target_f in zip(self.online_f.parameters(), self.target_f.parameters()):
            param_target_f.data = param_target_f.data * self.m + param_online_f.data * (1. - self.m)
        for param_online_g, param_target_g in zip(self.online_g.parameters(), self.target_g.parameters()):
            param_target_g.data = param_target_g.data * self.m + param_online_g.data * (1. - self.m)

    def forward(self, data):
        im_1 = data[0]
        im_2 = data[1]
        im_1 = im_1.to(device)
        im_2 = im_2.to(device)

        online_feature_1 = self.online_f(im_1)
        online_feature_2 = self.online_f(im_2)

        online_proj_1 = self.online_g(online_feature_1)
        online_proj_2 = self.online_g(online_feature_2)

        online_pred_1 = self.online_q(online_proj_1)
        online_pred_2 = self.online_q(online_proj_2)

        with torch.no_grad():
            self._momentum_update_target()

            target_feature_1 = self.target_f(im_1)
            target_feature_2 = self.target_f(im_2)

            target_proj_1 = self.target_g(target_feature_1)
            target_proj_2 = self.target_g(target_feature_2)

        loss_1 = loss_fn(online_pred_1, target_proj_2.detach())
        loss_2 = loss_fn(online_pred_2, target_proj_1.detach())

        loss = (loss_1 + loss_2).mean()
        return loss


# ========================    开始训练    ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Loss training with Pytorch")
    parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.config), args.config
    opt = SourceFileLoader('module.name', args.config).load_module().opt

    # ========================    数据读取    =========================
    read_train = opt.read_data.train
    from utils.transforms import transform

    train_trans, test_trans = transform()
    trainset = MyDataset(txt_path=opt.read_data.train.file_path, transform=test_trans)
    trainloader = DataLoader(trainset, batch_size=opt.read_data.train.batch_size, shuffle=False)
    testset = MyDataset(txt_path=opt.read_data.test.file_path, transform=test_trans)
    testloader = DataLoader(testset, batch_size=opt.read_data.test.batch_size, shuffle=False)

    # ========================    导入网络    ========================
    byol = BYOL(opt).to(device)
    byol.load_state_dict(torch.load('log/Res22_Side_super_byol-Dot-standard/net6499.pth'))  # 需修改
    print("模型导入成功！")

    # ========================   提取特征   =======================
    import numpy as np

    feature_extract = np.random.rand(50000, 128)
    byol.eval()
    with torch.no_grad():
        for i_iter, data in enumerate(trainloader):
            img, label = data
            img, label = img.to(opt.device), label.to(opt.device)

            x = byol.online_f(img)

            feature_extract[opt.read_data.train.batch_size * i_iter: opt.read_data.train.batch_size * (i_iter + 1),
            :] = x.cpu().data.numpy()

            print("当前提取进度：", i_iter)
        feature_extract = torch.from_numpy(feature_extract)

    # ========================   计算权重   =========================
    for i in range(1, 11):
        temp = feature_extract[(i - 1) * 5000: i * 5000, :]
        if i == 1:
            weight_mean = torch.mean(temp, dim=0).view(1, -1)
            weight_median = torch.median(temp, dim=0).values.view(1, -1)
        else:
            temp_mean = torch.mean(temp, dim=0).view(1, -1)
            weight_mean = torch.cat([weight_mean, temp_mean], dim=0)
            temp_median = torch.median(temp, dim=0).values.view(1, -1)
            weight_median = torch.cat([weight_median, temp_median], dim=0)

    # ========================  测试计算   ==========================
    mean_correct = 0
    median_correct = 0
    total = 0
    byol.eval()
    feature_test = np.random.rand(10000, 128)
    labels = []
    for i_iter, data in enumerate(testloader):
        img, label = data
        img, label = img.to(opt.device), label.to(opt.device)

        feature = byol.online_f(img)
        feature_test[opt.read_data.test.batch_size * i_iter: opt.read_data.test.batch_size * (i_iter + 1),
        :] = feature.cpu().data.numpy()
        labels.extend(label.cpu().data.numpy().tolist())

        mean_mm = torch.mm(feature.cpu().double(), weight_mean.t())
        median_mm = torch.mm(feature.cpu().double(), weight_median.t())

        _, mean_predicted = torch.max(mean_mm.data, dim=1)
        _, median_predicted = torch.max(median_mm.data, dim=1)

        mean_correct += mean_predicted.eq(label.cpu().data).cpu().sum().item()
        median_correct += median_predicted.eq(label.cpu().data).cpu().sum().item()
        total += label.size(0)
    print("mean_acc:", mean_correct / total)
    print("median_acc:", median_correct / total)

    feature_test = torch.from_numpy(feature_test)
    feature_test = F.normalize(feature_test, dim=-1, p=2)
    feature_test = feature_test.data.numpy()
    labels = np.array(labels)
    corr = np.dot(feature_test, feature_test.T)
    row, col = np.diag_indices_from(corr)
    corr[row, col] = 0
    index = np.argmax(corr, axis=1)
    # print(index[600:1100])
    # print(labels[600:1100])
    # print(feature_test[1, :])
    # print(feature_test[100, :])
    # print(feature_test[1560, :])
    for i in range(len(index)):
        if index[i] < 1000:
            index[i] = 0
        elif index[i] < 2000:
            index[i] = 1
        elif index[i] < 3000:
            index[i] = 2
        elif index[i] < 4000:
            index[i] = 3
        elif index[i] < 5000:
            index[i] = 4
        elif index[i] < 6000:
            index[i] = 5
        elif index[i] < 7000:
            index[i] = 6
        elif index[i] < 8000:
            index[i] = 7
        elif index[i] < 9000:
            index[i] = 8
        else:
            index[i] = 9
    corr_correct = sum(index == labels)
    print("corr_acc:", corr_correct / total)
