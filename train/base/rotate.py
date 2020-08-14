import torch
import torch.optim as optim
import os
import argparse
from importlib.machinery import SourceFileLoader

from utils.my_dataset import MyDataset, Rotate
from utils.transforms import transform
from torch.utils.data import DataLoader
from moduls.modul import Net
from moduls.loss import Loss
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
    train_trans, test_trans = transform()
    trainset = Rotate(txt_path=opt.read_data.train.file_path, transform=train_trans)
    trainloader = DataLoader(trainset, batch_size=opt.read_data.train.batch_size, shuffle=opt.read_data.train.shuffle)
    testset = MyDataset(txt_path=opt.read_data.test.file_path, transform=test_trans)
    testloader = DataLoader(testset, batch_size=opt.read_data.test.batch_size, shuffle=False)

    # ========================    导入网络    ========================
    net = Net(opt).to(opt.device)
    if opt.train.is_net_load:
        net.load_state_dict(torch.load(opt.train.net_path))
        print("模型导入成功！")
    criterion = Loss(opt).to(opt.device)
    fc = torch.nn.Linear(128, 4, bias=False).to(opt.device)
    criterion_rotate = torch.nn.CrossEntropyLoss().to(opt.device)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr_init, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)
    optimizer_rotate = optim.SGD(fc.parameters(), lr=opt.lr_init, momentum=0.9, weight_decay=0.0003)
    scheduler_rotate = optim.lr_scheduler.MultiStepLR(optimizer_rotate, opt.lr_mul, gamma=opt.lr_gamma)

    # ========================     设置log路径 ======================
    log = Log(opt)

    # ========================   训练及测试   =======================
    for i_epoch in range(opt.train.max_epoch):
        # --------------------   训练     -------------------------
        net.train()
        fc.train()
        optimizer.zero_grad()
        scheduler.step()
        optimizer_rotate.zero_grad()
        scheduler_rotate.step()
        log.init()
        trainloader_len = len(trainloader)
        for i_iter, data in enumerate(trainloader):
            img, label, label_rotate = data
            img, label, label_rotate = img.to(opt.device), label.to(opt.device), label_rotate.to(opt.device)

            optimizer.zero_grad()
            optimizer_rotate.zero_grad()
            x, feat = net(img, label, is_train=True)
            loss_info = criterion(x, label)
            x_rotate = fc(feat)
            loss_rotate = criterion_rotate(x_rotate, label_rotate)
            loss = (loss_info[0] + loss_rotate).mean()
            loss.backward()
            optimizer.step()
            optimizer_rotate.step()

            log.update(x, label)
            log.log_train(optimizer, loss_info, opt, i_epoch, i_iter, trainloader_len)

        # --------------------   测试     -------------------------
        print("测试...")
        with torch.no_grad():
            net.eval()
            fc.eval()
            log.init()
            for data in testloader:
                img, label = data
                img, label = img.to(opt.device), label.to(opt.device)

                x = net(img, label, is_train=False)[0]

                log.update(x, label)
            log.log_test(net, opt, i_epoch, trainloader_len)
    print("训练完成")
