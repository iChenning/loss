import torch.nn as nn


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        from moduls.modul_net5 import Net5
        from moduls.modul_resnet22 import ResNet22
        from moduls.modul_resnet26 import ResNet26
        from moduls.modul_ACRes26 import ACRes26
        from moduls.modul_fc_weight import Dot, Cos, CosAddMargin

        if opt.train.feature_net == 'Net5':
            self.feature_net = Net5()
        elif opt.train.feature_net == 'Resnet22':
            self.feature_net = ResNet22()
        elif opt.train.feature_net == 'Resnet26':
            self.feature_net = ResNet26()
        else:
            self.feature_net = ACRes26()

        self.fc_type = opt.train.fc_type
        if opt.train.fc_type == 'Cos':
            self.fc = Cos(opt)
        elif opt.train.fc_type == 'CosAddMargin':
            self.fc = CosAddMargin(opt)
        else:
            self.fc = Dot(opt)

    def forward(self, img, label, is_train=True):
        feature = self.feature_net(img)
        x = self.fc(feature, label, is_train=is_train)
        return (x, feature)