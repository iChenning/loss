import torch
import torch.nn as nn
import torch.nn.functional as F


class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        self.is_1x1 = kernel_size == 1

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode
            )
        else:
            if self.is_1x1:
                self.conv1x1_bn = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, out_channels=out_channels,
                        kernel_size=(1, 1), stride=stride, padding=padding,
                        dilation=dilation, groups=groups, bias=False,
                        padding_mode=padding_mode
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                )
            else:
                self.square_conv = nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding,
                    dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode
                )
                self.square_bn = nn.BatchNorm2d(num_features=out_channels)

                center_offset_from_origin_border = padding - kernel_size // 2
                ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
                hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
                if center_offset_from_origin_border >= 0:
                    self.ver_conv_crop_layer = nn.Identity()
                    ver_conv_padding = ver_pad_or_crop
                    self.hor_conv_crop_layer = nn.Identity()
                    hor_conv_padding = hor_pad_or_crop
                else:
                    self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                    ver_conv_padding = (0, 0)
                    self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                    hor_conv_padding = (0, 0)
                self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                          stride=stride,
                                          padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                          padding_mode=padding_mode)

                self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                          stride=stride,
                                          padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                          padding_mode=padding_mode)
                self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
                self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            if self.is_1x1:
                return self.conv1x1_bn(input)
            else:
                square_outputs = self.square_conv(input)
                square_outputs = self.square_bn(square_outputs)
                vertical_outputs = self.ver_conv_crop_layer(input)
                vertical_outputs = self.ver_conv(vertical_outputs)
                vertical_outputs = self.ver_bn(vertical_outputs)
                horizontal_outputs = self.hor_conv_crop_layer(input)
                horizontal_outputs = self.hor_conv(horizontal_outputs)
                horizontal_outputs = self.hor_bn(horizontal_outputs)
                return square_outputs + vertical_outputs + horizontal_outputs


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            ACBlock(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            ACBlock(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
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


class ACRes(nn.Module):
    def __init__(self, ResidualBlock, num_classes=128):
        super(ACRes, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.layer5 = self.make_layer(ResidualBlock, 1024, 2, stride=2)
        self.layer6 = self.make_layer(ResidualBlock, 1024, 2, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(1024, num_classes, bias=False),
            nn.ReLU(inplace=True)
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
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
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def ACRes26():
    return ACRes(ResidualBlock)

