import torch
import torch.optim as optim
import os
import argparse
from importlib.machinery import SourceFileLoader

from utils.my_dataset import dataloader
from moduls.modul import Net
from moduls.loss import Loss

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
    criterion = Loss(opt).to(opt.device)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)

    # ========================     设置log路径 ======================
    from utils.log import Log

    log = Log(opt)

    # ========================   训练及测试   =======================
    best_acc = 0.60
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
            x = net(img, label, is_train=True)
            loss_info = criterion(x, label)
            loss = loss_info[0]
            loss.backward()
            optimizer.step()

            log.update(x, label)
            log.log_train(scheduler, loss_info, opt, i_epoch, i_iter, trainloader_len)

        # --------------------   测试     -------------------------
        print("测试...")
        with torch.no_grad():
            net.eval()
            log.init()
            for data in testloader:
                img, label = data
                img, label = img.to(opt.device), label.to(opt.device)

                x = net(img, label, is_train=False)

                log.update(x, label)
            log.log_test(net, opt, i_epoch, trainloader_len)
    print("训练完成")
