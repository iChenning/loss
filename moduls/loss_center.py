import torch
import torch.nn as nn


class Center(nn.Module):
    def __init__(self, opt):
        super(Center, self).__init__()
        self.scale = opt.train.scale
        self.device = opt.device

    def forward(self, feature, label):
        one_hot = torch.zeros(feature.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        feature_s = feature * one_hot / self.scale
        # loss_center = (torch.sum(torch.sqrt(torch.tensor(2.) - feature_s * torch.tensor(2.))) - torch.sqrt(
        #     torch.tensor(2.)) * (feature.size(0) * feature.size(1) - feature.size(0))) / feature.size(0)
        loss_center = 2.0 - 2.0 * torch.sum(feature_s) / label.size(0)

        return loss_center
