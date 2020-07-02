import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LossAddCenter(nn.Module):
    def __init__(self):
        super(LossAddCenter, self).__init__()

    def forward(self, feature, label):
        one_hot = torch.zeros(feature.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        feature_s = feature * one_hot * 0.01
        loss_center = (torch.sum(torch.sqrt(torch.tensor(2.) - feature_s * torch.tensor(2.))) - torch.sqrt(
            torch.tensor(2.)) * (feature.size(0) * feature.size(1) - feature.size(0))) / feature.size(0)

        return loss_center