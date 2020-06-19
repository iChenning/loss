import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

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


class AddMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=10, s=100.0, m=0.02):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, epoch):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine = F.linear(input, self.weight)
        phi = cosine - self.m*(epoch % 10 +1)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'