import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, opt, num_classes=128):
        super(ResNet, self).__init__()
        self.is_side1 = opt.is_side1
        self.side1 = self.__side_net(3, win_size=3, is_bn=False, is_act=True)
        self.inchannel = 64
        self.conv1 = nn.Sequential(  # 64*32*32
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)  # 64*32*32
        self.is_side2 = opt.is_side2
        self.side2 = self.__side_net(64, win_size=3, is_bn=False, is_act=True)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)  # 128*16*16
        self.is_side3 = opt.is_side3
        self.side3 = self.__side_net(128, win_size=3, is_bn=False, is_act=True)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)  # 256*8*8
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)  # 512*4*4
        self.avgPool = nn.AdaptiveAvgPool2d((2, 2))  # 512*2*2
        self.fc = nn.Sequential(
            nn.Linear(512*2*2, num_classes, bias=False),
            nn.ReLU(inplace=True)
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.is_side1:
            x = self.side1(x)
        out = self.conv1(x)
        out = self.layer1(out)
        if self.is_side2:
            out = self.side2(out)
        out = self.layer2(out)
        if self.is_side3:
            out = self.side3(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def __side_net(self, channels, win_size=3, is_bn=False, is_act=True):
        from .modul_side import Side
        side = Side(channels, win_size, is_bn, is_act)
        return side

def SideResNet18(opt):
    print("new!!!")
    return ResNet(ResidualBlock, opt)
