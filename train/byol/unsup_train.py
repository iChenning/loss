from utils.my_dataset import unsup_BYOL, MyDataset
from utils.transforms import transform_unsuper
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
from moduls.modul_fc_weight import Dot, Cos, CosAddMargin

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
    train_trans, test_trans = transform_unsuper()
    trainset = unsup_BYOL(txt_path=opt.read_data.train.file_path, transform=train_trans)
    trainloader = DataLoader(trainset, batch_size=opt.read_data.train.batch_size, shuffle=opt.read_data.train.shuffle)
    testset = MyDataset(txt_path=opt.read_data.test.file_path, transform=test_trans)
    testloader = DataLoader(testset, batch_size=opt.read_data.test.batch_size, shuffle=False)

    # ========================    导入网络    ========================
    byol = BYOL(opt).to(device)

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(byol.parameters(), lr=opt.lr_init, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)

    # ========================     训练日志 ==========================
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    if opt.log_name == None:
        log_dir = os.path.join('log', opt.train.feature_net + '-' + opt.train.fc_type + '-' + opt.train.loss_type + '_' + time_str)
    else:
        log_dir = os.path.join('log', opt.train.feature_net + '_' + opt.log_name + '-' + opt.train.fc_type + '-' + opt.train.loss_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ========================   训练   =======================
    best_acc = 0.60
    for i_epoch in range(opt.train.max_epoch):
        byol.train()
        optimizer.zero_grad()
        scheduler.step()
        for i_iter, data in enumerate(trainloader):

            optimizer.zero_grad()

            loss = byol(data)
            loss.backward()
            optimizer.step()
            print("Training: Epoch[{:0>3}/{:0>3}] ""Iteration[{:0>3}/{:0>3}] ""Loss: {:.4f} ".format(
                i_epoch + 1, opt.train.max_epoch, i_iter + 1, len(trainloader), loss.item()))
            writer.add_scalars('Loss_group', {'train_loss': loss.item()}, i_epoch * len(trainloader) + i_iter)
            writer.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], i_epoch * len(trainloader) + i_iter)

        if (i_epoch + 1) % 50 == 0:
            torch.save(byol.state_dict(), '%s/net%d.pth' % (log_dir, i_epoch))
