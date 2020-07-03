import torch.nn as nn
from moduls.loss_center import Center

class Loss(nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()

        if opt.train.fc_type == 'Dot' and opt.train.loss_type == 'add_center':
            assert False, "Dot and center can not use together, center must used with cos!!!"

        self.is_add_center = True if opt.train.loss_type == 'add_center' else False
        if self.is_add_center:
            self.criterion_1 = nn.CrossEntropyLoss()
            self.criterion_2 = Center(opt)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        if self.is_add_center:
            loss_1 = self.criterion_1(x, label)
            loss_2 = self.criterion_2(x, label)
            loss = (loss_1 + loss_2) / 2.0
            return (loss, loss_1, loss_2)
        else:
            loss = self.criterion(x, label)
            return (loss,)
