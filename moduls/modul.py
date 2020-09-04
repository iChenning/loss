import torch.nn as nn
import torchvision
import torch


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        from moduls.modul_net5 import Net5
        from moduls.modul_SideNet5 import SideNet5
        from moduls.modul_Side123Res18 import SideResNet18
        from moduls.modul_SideRes18 import SideRes18
        from moduls.modul_resnet22 import ResNet22
        from moduls.modul_SideRes22 import SideResNet22
        from moduls.modul_resnet26 import ResNet26
        from moduls.modul_ACRes26 import ACRes26
        from moduls.modul_fc_weight import Dot, Cos, CosAddMargin

        if opt.train.feature_net == 'Net5':
            self.feature_net = Net5()
        elif opt.train.feature_net == 'Net5_Side':
            self.feature_net = SideNet5(opt)
        elif opt.train.feature_net == 'Res18_Side123':
            self.feature_net = SideResNet18(opt)
        elif opt.train.feature_net == 'Res18_SideNew':
            self.feature_net = SideRes18()
        elif opt.train.feature_net == 'Resnet22':
            self.feature_net = ResNet22()
        elif opt.train.feature_net == 'Res22_Side':
            self.feature_net = SideResNet22(opt)
        elif opt.train.feature_net == 'Resnet26':
            self.feature_net = ResNet26()
        else:
            self.feature_net = ACRes26()

        feature_net = torchvision.models.resnet50()
        feature_net = nn.Sequential(*list(feature_net.children())[: -2])
        feature_net.add_module('drop_out', nn.Dropout(p=0.4))
        self.feature_net = feature_net
        self.feature = nn.Linear(2048 * 7 * 7, 128, bias=False)
        self.bn = nn.BatchNorm1d(128)


        self.fc_type = opt.train.fc_type
        if opt.train.fc_type == 'Cos':
            self.fc = Cos(opt)
        elif opt.train.fc_type == 'CosAddMargin':
            self.fc = CosAddMargin(opt)
        else:
            self.fc = Dot(opt)

    def forward(self, img, label, is_train=True):
        feature = self.feature_net(img)

        feature = torch.flatten(feature, 1)
        feature = self.feature(feature)
        feature = self.bn(feature)

        x = self.fc(feature, label, is_train=is_train)
        return (x, feature)
