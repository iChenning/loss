import torch
import torch.nn as nn
import torch.nn.functional as F


class SideNet5(nn.Module):
    def __init__(self):
        super(SideNet5, self).__init__()
        self.side1 = self.__side_net(3, win_size=3, is_bn=False, is_act=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.side2 = self.__side_net(64, win_size=3, is_bn=False, is_act=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.side3 = self.__side_net(128, win_size=3, is_bn=False, is_act=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.side1(x)
        x = self.conv1(x)
        x = self.side2(x)
        x = self.conv2(x)
        x = self.side3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)

        return x

    def __side_net(self, channels, win_size=3, is_bn=False, is_act=True):
        from .modul_side import Side
        side = Side(channels, win_size, is_bn, is_act)
        return side