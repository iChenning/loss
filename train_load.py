from utils.my_dataset import MyDataset2, data_prefetcher
import torchvision
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
from moduls.modul_net5 import Net5
from moduls.modul_resnet22 import ResNet22
from moduls.modul_resnet26 import ResNet26
from moduls.modul_ACRes26 import ACRes26
from moduls.modul_fc_weight import Dot, Cos, CosAddMargin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
if torch.cuda.device_count() >= 2:
    torch.cuda.set_device(3)  # os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOL(nn.Module):
    def __init__(self, opt, m=0.996):
        super(BYOL, self).__init__()

        self.m = m

        if opt.train.feature_net == 'Net5':
            self.online_f = Net5(opt)
            self.target_f = Net5(opt)  # copy.deepcopy(self.online_f)
        elif opt.train.feature_net == 'Resnet22':
            self.online_f = ResNet22()
            self.target_f = ResNet22()
        elif opt.train.feature_net == 'Resnet26':
            self.online_f = ResNet26()
            self.target_f = ResNet26()
        else:
            self.online_f = ACRes26()
            self.target_f = ACRes26()

        for param_online_f, param_target_f in zip(self.online_f.parameters(), self.target_f.parameters()):
            param_target_f.data.copy_(param_online_f.data)  # initialize
            param_target_f.requires_grad = False  # not update by gradient

        dim_mlp = 128
        self.online_g = nn.Sequential(nn.Linear(dim_mlp, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, dim_mlp))
        self.target_g = nn.Sequential(nn.Linear(dim_mlp, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(1024, dim_mlp))
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

        with torch.no_grad():  # no gradient to keys
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
    trainset = MyDataset2(txt_path=read_train.file_path, transform=read_train.transforms)
    trainloader = DataLoader(trainset, batch_size=read_train.batch_size, shuffle=read_train.shuffle)

    # ========================    导入网络    ========================
    byol = BYOL(opt).to(device)
    byol.load_state_dict(torch.load("./model/net_020.pth"))

    # ========================    初始化优化器 =======================
    optimizer = optim.SGD(byol.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_mul, gamma=opt.lr_gamma)
    c = nn.CrossEntropyLoss()

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join('log', time_str + '_' + opt.train.feature_net + '-' + opt.train.fc_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ========================   训练及测试   =======================
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

            print("Training: Epoch[{:0>3}/{:0>3}] "
                  "Iteration[{:0>3}/{:0>3}] "
                  "Loss: {:.4f} ".format(
                i_epoch + 1, opt.train.max_epoch,
                i_iter + 1, len(trainloader),
                loss.item())
            )

            writer.add_scalars('Loss_group', {'train_loss': loss.item()},
                               i_epoch * len(trainloader) + i_iter)
            writer.add_scalar('learning rate', scheduler.get_lr()[0], i_epoch * len(trainloader) + i_iter)

        print('Saving model......')
        torch.save(byol.state_dict(), '%s/net_%03d.pth' % (opt.module_save.path, i_epoch + 1))

    print("训练完成")
