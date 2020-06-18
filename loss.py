import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, s=None, m=None):
        super(Loss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, x, label):
        assert len(x) == len(label), "样本维度和label长度不一致"

        numerator = self.s * (torch.diagonal(x.transpose(0, 1)[label]) - self.m)

        excl = torch.cat([torch.cat((x[i, :y], x[i, y + 1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = (numerator - torch.log(denominator))/self.s
        return -torch.mean(L)