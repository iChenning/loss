import torch
import os
import argparse
from importlib.machinery import SourceFileLoader
import numpy as np

from utils.my_dataset import dataloader
from moduls.modul import Net

# ========================    开始训练    ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Loss training with Pytorch")
    parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.config), args.config
    opt = SourceFileLoader('module.name', args.config).load_module().opt

    # ========================    数据读取    =========================
    trainloader, testloader = dataloader(opt)

    # ========================    导入网络    ========================
    net = Net(opt).to(opt.device)
    net.load_state_dict(torch.load("./model/net_175.pth"))
    # for name,pars in net.named_parameters():
    #     print(name,pars)

    # ========================   提取特征   =======================
    feature_extract = np.random.rand(50000, 128)
    for i_iter, data in enumerate(trainloader):
        img, label = data
        img, label = img.to(opt.device), label.to(opt.device)

        x = net(img, label, is_train=False)[1]

        feature_extract[opt.read_data.train.batch_size * i_iter : opt.read_data.train.batch_size * (i_iter + 1), :] = x.cpu().data.numpy()

        # if i_iter == 0:
        #     feature_extract = x
        #     labels = label
        # else:
        #     feature_extract = torch.cat([feature_extract, x], dim=0)
        #     labels = torch.cat([labels, label], dim = 0)

        print("当前提取进度：", i_iter)
    feature_extract = torch.from_numpy(feature_extract)

    # ========================   计算权重   =========================
    for i in range(1,11):
        temp = feature_extract[(i - 1) * 5000 : i * 5000, :]
        if i == 1:
            weight_mean = torch.mean(temp, dim=0).view(1,-1)
            weight_median = torch.median(temp, dim=0).values.view(1,-1)
        else:
            temp_mean = torch.mean(temp, dim=0).view(1,-1)
            weight_mean = torch.cat([weight_mean, temp_mean], dim=0)
            temp_median = torch.median(temp, dim=0).values.view(1,-1)
            weight_median = torch.cat([weight_median, temp_median], dim=0)
    # print("mean:")
    # print(weight_mean)
    # print("median:")
    # print(weight_median)

    # ========================  测试计算   ==========================
    mean_correct = 0
    median_correct = 0
    origin_correct = 0
    total = 0
    for i_iter, data in enumerate(testloader):
        img, label = data
        img, label = img.to(opt.device), label.to(opt.device)

        x, feature = net(img, label, is_train=False)

        mean_mm = torch.mm(feature.cpu().double(), weight_mean.t())
        median_mm = torch.mm(feature.cpu().double(), weight_median.t())

        _, mean_predicted = torch.max(mean_mm.data, dim=1)
        _, median_predicted = torch.max(median_mm.data, dim=1)
        _, origin_predicted = torch.max(x.data, dim=1)

        mean_correct += mean_predicted.eq(label.cpu().data).cpu().sum().item()
        median_correct += median_predicted.eq(label.cpu().data).cpu().sum().item()
        origin_correct += origin_predicted.eq(label.data).cpu().sum().item()
        total += label.size(0)
    print("mean_acc:", mean_correct / total)
    print("median_acc:", median_correct / total)
    print("origin_acc:", origin_correct / total)
