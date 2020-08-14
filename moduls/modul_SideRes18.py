import torch
import torch.nn as nn
import torch.nn.functional as F


class Side(nn.Module):
    def __init__(self, inchannel, outchannel, win_size=3, stride=1, is_bn=False, is_act=True):
        super(Side, self).__init__()
        self.is_bn = is_bn
        self.is_act = is_act

        pad_size = win_size // 2
        self.pad_all = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
        self.pad_2_1 = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, 0))
        self.pad_2_2 = nn.ReplicationPad2d(padding=(0, pad_size, pad_size, pad_size))
        self.pad_2_3 = nn.ReplicationPad2d(padding=(pad_size, pad_size, 0, pad_size))
        self.pad_2_4 = nn.ReplicationPad2d(padding=(pad_size, 0, pad_size, pad_size))
        self.pad_1_1 = nn.ReplicationPad2d(padding=(pad_size, 0, pad_size, 0))
        self.pad_1_2 = nn.ReplicationPad2d(padding=(0, pad_size, pad_size, 0))
        self.pad_1_3 = nn.ReplicationPad2d(padding=(0, pad_size, 0, pad_size))
        self.pad_1_4 = nn.ReplicationPad2d(padding=(pad_size, 0, 0, pad_size))

        self.conv_all = nn.Conv2d(inchannel, outchannel, kernel_size=(win_size, win_size), stride=stride, bias=False)
        self.conv_2_1 = nn.Conv2d(inchannel, outchannel, kernel_size=(pad_size + 1, win_size), stride=stride, bias=False)
        self.conv_2_2 = nn.Conv2d(inchannel, outchannel, kernel_size=(win_size, pad_size + 1), stride=stride, bias=False)
        self.conv_2_3 = nn.Conv2d(inchannel, outchannel, kernel_size=(pad_size + 1, win_size), stride=stride, bias=False)
        self.conv_2_4 = nn.Conv2d(inchannel, outchannel, kernel_size=(win_size, pad_size + 1), stride=stride, bias=False)
        self.conv_1_1 = nn.Conv2d(inchannel, outchannel, kernel_size=(pad_size + 1, pad_size + 1), stride=stride, bias=False)
        self.conv_1_2 = nn.Conv2d(inchannel, outchannel, kernel_size=(pad_size + 1, pad_size + 1), stride=stride, bias=False)
        self.conv_1_3 = nn.Conv2d(inchannel, outchannel, kernel_size=(pad_size + 1, pad_size + 1), stride=stride, bias=False)
        self.conv_1_4 = nn.Conv2d(inchannel, outchannel, kernel_size=(pad_size + 1, pad_size + 1), stride=stride, bias=False)

        self.att = nn.Conv2d(outchannel * 9, outchannel, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_all = self.pad_all(x)
        x_all = self.conv_all(x_all)

        x_21 = self.pad_2_1(x)
        x_21 = self.conv_2_1(x_21)

        x_22 = self.pad_2_2(x)
        x_22 = self.conv_2_2(x_22)

        x_23 = self.pad_2_3(x)
        x_23 = self.conv_2_3(x_23)

        x_24 = self.pad_2_4(x)
        x_24 = self.conv_2_4(x_24)

        x_11 = self.pad_1_1(x)
        x_11 = self.conv_1_1(x_11)

        x_12 = self.pad_1_2(x)
        x_12 = self.conv_1_2(x_12)

        x_13 = self.pad_1_3(x)
        x_13 = self.conv_1_3(x_13)

        x_14 = self.pad_1_4(x)
        x_14 = self.conv_1_4(x_14)

        x_cat = torch.cat([x_all, x_21, x_22, x_23, x_24, x_11, x_12, x_13, x_14], dim=1)
        out = self.att(x_cat)
        if self.is_bn:
            out = self.bn(out)
        if self.is_act:
            out = self.act(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left1 = Side(inchannel, outchannel, win_size=3, stride=stride, is_bn=True, is_act=True)
        self.left2 = Side(outchannel, outchannel, win_size=3, stride=1, is_bn=True, is_act=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left1(x)
        out = self.left2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, feat_dim=128):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = Side(3, 64, win_size=7, stride=1, is_bn=True, is_act=True)  # 64*32*32
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)  # 64*32*32
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)  # 128*16*16
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)  # 256*8*8
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)  # 512*4*4
        self.avgPool = nn.AdaptiveAvgPool2d((2, 2))  # 512*2*2
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, feat_dim, bias=False),
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
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def SideRes18():
    return ResNet(ResidualBlock)


