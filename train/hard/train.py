import torch
import torch.optim as optim
import os
import argparse
from importlib.machinery import SourceFileLoader

from utils.my_dataset import dataloader
from moduls.modul import Net
from moduls.loss import Loss

import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

# ========================    开始训练    ========================
if __name__ == "__main__":
    print("当前工作路径为:", os.getcwd())
    parser = argparse.ArgumentParser("Loss training with Pytorch")
    parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.config), args.config
    opt = SourceFileLoader('module.name', args.config).load_module().opt

    # ========================    数据读取    =========================
    trainloader, testloader = dataloader(opt)

    # ========================    导入网络    ========================
    net = Net(opt).to(opt.device)
    if opt.train.is_net_load:
        net.load_state_dict(torch.load(opt.train.net_path))
        print("模型导入成功！")
    criterion = Loss(opt).to(opt.device)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr_init, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)

    # ========================    提取分数    =======================
    score_trian = np.random.rand(50000, 10)
    score_test = np.random.rand(10000, 10)

    with torch.no_grad():
        net.eval()
        for i_iter, data in enumerate(trainloader):
            img, label = data
            img, label = img.to(opt.device), label.to(opt.device)

            optimizer.zero_grad()
            x = net(img, label, is_train=True)[0]
            score_trian[opt.read_data.train.batch_size * i_iter: opt.read_data.train.batch_size * (i_iter + 1),
            :] = x.cpu().data.numpy()

        for i_iter, data in enumerate(testloader):
            img, label = data
            img, label = img.to(opt.device), label.to(opt.device)

            x = net(img, label, is_train=False)[0]
            score_test[opt.read_data.test.batch_size * i_iter: opt.read_data.test.batch_size * (i_iter + 1),
            :] = x.cpu().data.numpy()

            _, predicted = torch.max(x.data, 1)
            index = np.where(predicted.eq(label.data).cpu().data.numpy() == False)[0]
            print(index + i_iter * opt.read_data.test.batch_size)

    # =========================    分数方差    ========================
    score_train_t = torch.from_numpy(score_trian)
    score_train_t = F.softmax(score_train_t, dim=1)
    score_trian = score_train_t.data.numpy()
    score_test_t = torch.from_numpy(score_test)
    score_test_t = F.softmax(score_test_t, dim=1)
    score_test = score_test_t.data.numpy()
    std_train = np.std(score_trian, axis=1)
    std_test = np.std(score_test, axis=1)
    print("ceshi:",np.where(std_test <= 0.175)[0])
    print(score_test[176, :])
    print(score_test[399, :])
    print(score_test[400, :])
    fig1 = plt.figure(1)
    plt.plot(std_train)
    fig1.savefig("std_train.png")
    fig2 = plt.figure(2)
    plt.plot(std_test)
    fig2.savefig("std_test.png")
