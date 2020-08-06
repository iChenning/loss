import torch
import torch.nn as nn


class Side(nn.Module):
    def __init__(self, channel, win_size=3, is_bn=False, is_act=True):
        super(Side, self).__init__()
        self.is_bn = is_bn
        self.is_act = is_act

        pad_size = win_size // 2
        self.pad_all = nn.ZeroPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
        self.pad_2_1 = nn.ZeroPad2d(padding=(pad_size, pad_size, pad_size, 0))
        self.pad_2_2 = nn.ZeroPad2d(padding=(0, pad_size, pad_size, pad_size))
        self.pad_2_3 = nn.ZeroPad2d(padding=(pad_size, pad_size, 0, pad_size))
        self.pad_2_4 = nn.ZeroPad2d(padding=(pad_size, 0, pad_size, pad_size))
        self.pad_1_1 = nn.ZeroPad2d(padding=(pad_size, 0, pad_size, 0))
        self.pad_1_2 = nn.ZeroPad2d(padding=(0, pad_size, pad_size, 0))
        self.pad_1_3 = nn.ZeroPad2d(padding=(0, pad_size, 0, pad_size))
        self.pad_1_4 = nn.ZeroPad2d(padding=(pad_size, 0, 0, pad_size))

        self.conv_all = nn.Conv2d(channel, channel, kernel_size=(win_size, win_size), stride=1, bias=False)
        self.conv_2_1 = nn.Conv2d(channel, channel, kernel_size=(pad_size + 1, win_size), stride=1, bias=False)
        self.conv_2_2 = nn.Conv2d(channel, channel, kernel_size=(win_size, pad_size + 1), stride=1, bias=False)
        self.conv_2_3 = nn.Conv2d(channel, channel, kernel_size=(pad_size + 1, win_size), stride=1, bias=False)
        self.conv_2_4 = nn.Conv2d(channel, channel, kernel_size=(win_size, pad_size + 1), stride=1, bias=False)
        self.conv_1_1 = nn.Conv2d(channel, channel, kernel_size=(pad_size + 1, pad_size + 1), stride=1, bias=False)
        self.conv_1_2 = nn.Conv2d(channel, channel, kernel_size=(pad_size + 1, pad_size + 1), stride=1, bias=False)
        self.conv_1_3 = nn.Conv2d(channel, channel, kernel_size=(pad_size + 1, pad_size + 1), stride=1, bias=False)
        self.conv_1_4 = nn.Conv2d(channel, channel, kernel_size=(pad_size + 1, pad_size + 1), stride=1, bias=False)

        self.att = nn.Conv2d(channel * 8, channel, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.act = nn.ReLU()

    def forward(self, x):
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

        x_cat = torch.cat([x_21, x_22, x_23, x_24, x_11, x_12, x_13, x_14], dim=1)
        out = self.att(x_cat)
        if self.is_bn:
            out = self.bn(out)
        if self.is_act:
            out = self.act(out)
        out += x

        return out
