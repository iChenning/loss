import torch
import torch.optim as optim
import os
import argparse
from importlib.machinery import SourceFileLoader

from utils.my_dataset import dataloader
from moduls.modul import Net
from moduls.loss import Loss
from utils.log import Log

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
    opt.train.fc_type = 'CosAddMargin'
    opt.train.scale = 100.0
    opt.train.margin = 0.2
    net2 = Net(opt).to(opt.device)
    criterion1 = Loss(opt).to(opt.device)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0003)
    optimizer2 = optim.SGD(net2.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, opt.lr_mul, gamma=opt.lr_gamma)

    # ========================     设置log路径 ======================
    log = Log(opt)

    # ========================   训练及测试   =======================
    for i_epoch in range(opt.train.max_epoch):
        # --------------------   训练     -------------------------
        net.train()
        net2.train()
        optimizer.zero_grad()
        optimizer2.zero_grad()
        scheduler.step()
        scheduler2.step()
        log.init()
        trainloader_len = len(trainloader)
        for i_iter, data in enumerate(trainloader):
            img, label = data
            img, label = img.to(opt.device), label.to(opt.device)

            optimizer.zero_grad()
            optimizer2.zero_grad()
            if i_epoch < 200:
                x = net(img, label, is_train=True)[0]
            else:
                if i_epoch == 200:
                    for param, param2 in zip(net.parameters(), net2.parameters()):
                        param2.data = param.data
                x = net2(img, label, is_train=True)[0]
            loss_info = criterion1(x, label)
            loss = loss_info[0]
            loss.backward()
            optimizer.step()
            optimizer2.step()

            log.update(x, label)
            log.log_train(scheduler, loss_info, opt, i_epoch, i_iter, trainloader_len)

        # --------------------   测试     -------------------------
        print("测试...")
        with torch.no_grad():
            net.eval()
            net2.eval()
            log.init()
            for data in testloader:
                img, label = data
                img, label = img.to(opt.device), label.to(opt.device)
                if i_epoch < 200:
                    x = net(img, label, is_train=False)[0]
                else:
                    x = net2(img, label, is_train=False)[0]

                log.update(x, label)
            log.log_test(net, opt, i_epoch, trainloader_len)
    print("训练完成")
