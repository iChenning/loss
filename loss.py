import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AddMarginLinear(nn.Module):
    def __init__(self, in_features=128, out_features=10, s=10.0, m=0.02):
        super(AddMarginLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)).to(device)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, epoch, is_train=True):
        assert len(input) == len(label), "样本维度和label长度不一致"

        if is_train:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            phi = cosine - self.m * (epoch % 10 + 1)
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + (
                        (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
        else:
            output = F.linear(F.normalize(input), F.normalize(self.weight))

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
