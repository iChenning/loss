import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


class Modules(nn.Module):
    def __init__(self, opt):
        super(Modules, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
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
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class Res5(nn.Module):
    def __init__(self, opt):
        super(Res5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.act = nn.ReLU(inplace=True)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(128, 10, bias=False)


    def forward(self, x):
        identity = self.shortcut1(x)
        x = self.conv1(x)
        x += identity
        x = self.act(x)

        identity = self.shortcut2(x)
        x = self.conv2(x)
        x += identity
        x = self.act(x)

        identity = self.shortcut3(x)
        x = self.conv3(x)
        x += identity
        x = self.act(x)

        identity = self.shortcut4(x)
        x = self.conv4(x)
        x += identity
        x = self.act(x)

        identity = self.shortcut5(x)
        x = self.conv5(x)
        x += identity
        x = self.act(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
