import torch
import torch.optim as optim
import os
import argparse
from importlib.machinery import SourceFileLoader

from utils.my_dataset import dataloader
from moduls.modul_net5 import Net5
from moduls.modul_SideNet5 import SideNet5
from moduls.modul_resnet22 import ResNet22
from moduls.modul_SideRes22 import SideResNet22
from moduls.modul_resnet26 import ResNet26
from moduls.modul_ACRes26 import ACRes26
from moduls.modul_fc_weight import Dot, Cos, CosAddMargin
from utils.log import Log

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
    if opt.train.feature_net == 'Net5':
        encoder = Net5().to(opt.device)
    elif opt.train.feature_net == 'Net5_Side':
        encoder = SideNet5(opt).to(opt.device)
    elif opt.train.feature_net == 'Resnet22':
        encoder = ResNet22().to(opt.device)
    elif opt.train.feature_net == 'Res22_Side':
        encoder = SideResNet22(opt).to(opt.device)
    elif opt.train.feature_net == 'Resnet26':
        encoder = ResNet26().to(opt.device)
    else:
        encoder = ACRes26().to(opt.device)

    if opt.train.fc_type == 'Cos':
        fc = Cos(opt)
    elif opt.train.fc_type == 'CosAddMargin':
        fc = CosAddMargin(opt)
    else:
        fc = Dot(opt)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr_init, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)

    # ========================     设置log路径 ======================
    log = Log(opt)

    # ========================   训练及测试   =======================
    for i_epoch in range(opt.train.max_epoch):
        # --------------------   训练     -------------------------
        net.train()
        optimizer.zero_grad()
        scheduler.step()
        log.init()
        trainloader_len = len(trainloader)
        for i_iter, data in enumerate(trainloader):
            img, label = data
            img, label = img.to(opt.device), label.to(opt.device)

            optimizer.zero_grad()
            x = net(img, label, is_train=True)[0]
            loss_info = criterion(x, label)
            loss = loss_info[0]
            loss.backward()
            optimizer.step()

            log.update(x, label)
            log.log_train(optimizer, loss_info, opt, i_epoch, i_iter, trainloader_len)

        # --------------------   测试     -------------------------
        print("测试...")
        with torch.no_grad():
            net.eval()
            log.init()
            for data in testloader:
                img, label = data
                img, label = img.to(opt.device), label.to(opt.device)

                x = net(img, label, is_train=False)[0]

                log.update(x, label)
            log.log_test(net, opt, i_epoch, trainloader_len)
    print("训练完成")
