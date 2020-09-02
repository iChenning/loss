import torch
import torch.optim as optim
import os
import argparse
from importlib.machinery import SourceFileLoader

from utils.transforms import normal_transforms
from utils.my_dataset import MyDataset
from torch.utils.data import DataLoader
from moduls.modul import Net
from moduls.loss import Loss
from utils.log import Log

if __name__ == "__main__":
    print("当前工作路径为:", os.getcwd())
    parser = argparse.ArgumentParser("Loss training with Pytorch")
    parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.config), args.config
    opt = SourceFileLoader('module.name', args.config).load_module().opt

    # ========================    数据读取    =========================
    train_trans, test_trans = normal_transforms()
    trainset = MyDataset(opt.data.train.file_path, transform=train_trans)
    trainloader = DataLoader(trainset, batch_size=opt.data.train.batch_size, shuffle=opt.data.train.shuffle)
    validset = MyDataset(opt.data.valid.file_path, transform=test_trans)
    validloader = DataLoader(validset, batch_size=opt.data.valid.batch_size, shuffle=opt.data.valid.shuffle)
    testset = MyDataset(opt.data.test.file_path, transform=test_trans)
    testloader = DataLoader(testset, batch_size=opt.data.test.batch_size, shuffle=opt.data.test.shuffle)

    # ========================    导入网络    ========================
    net = Net(opt).to(opt.device)
    if opt.train.is_net_load:
        net.load_state_dict(torch.load(opt.train.net_path))
        print("模型导入成功！")
    criterion = Loss(opt).to(opt.device)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(net.parameters(), lr=opt.train.lr_init, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.train.lr_mul, gamma=opt.train.lr_gamma)

    # ========================     设置log路径 ======================
    log = Log(opt)

    # ========================   训练及测试   =======================
    for i_epoch in range(opt.train.max_epoch):
        # ====================   导入最好模型 =======================
        if i_epoch + 1 in opt.train.lr_mul:
            if os.path.exists('%s/best_net.pth' % (log.log_dir)):
                net.load_state_dict(torch.load('%s/best_net.pth' % (log.log_dir)))
                print("模型导入成功！")

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

        # --------------------   验证     -------------------------
        print("验证...")
        with torch.no_grad():
            net.eval()
            log.init()
            for data in validloader:
                img, label = data
                img, label = img.to(opt.device), label.to(opt.device)

                x = net(img, label, is_train=False)[0]

                log.update(x, label)
            log.log_test(net, opt, i_epoch, trainloader_len)
    print("训练完成")

    # -----------------------   测试   ---------------------------
    if os.path.exists('%s/best_net.pth' % (log.log_dir)):
        net.load_state_dict(torch.load('%s/best_net.pth' % (log.log_dir)))
        print("模型导入成功！")
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