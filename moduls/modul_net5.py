import torch
import torch.nn as nn
import torch.nn.functional as F


# class Net5(nn.Module):
#     def __init__(self):
#         super(Net5, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(1024, 128, bias=False),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         # x = F.normalize(x, p=2, dim=1)
#
#         return x

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.side_all = nn.Conv2d(3, 3, kernel_size=3, stride=1, bias=False)
        self.ZeroPad_all = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.side_21 = nn.Conv2d(3, 3, kernel_size=(2, 3), stride=1, bias=False)
        self.ZeroPad_21 = nn.ZeroPad2d(padding=(1, 1, 1, 0))
        self.side_22 = nn.Conv2d(3, 3, kernel_size=(3, 2), stride=1, bias=False)
        self.ZeroPad_22 = nn.ZeroPad2d(padding=(0, 1, 1, 1))
        self.side_23 = nn.Conv2d(3, 3, kernel_size=(2, 3), stride=1, bias=False)
        self.ZeroPad_23 = nn.ZeroPad2d(padding=(1, 1, 0, 1))
        self.side_24 = nn.Conv2d(3, 3, kernel_size=(3, 2), stride=1, bias=False)
        self.ZeroPad_24 = nn.ZeroPad2d(padding=(1, 0, 1, 1))
        self.side_11 = nn.Conv2d(3, 3, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_11 = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.side_12 = nn.Conv2d(3, 3, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_12 = nn.ZeroPad2d(padding=(0, 1, 1, 0))
        self.side_13 = nn.Conv2d(3, 3, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_13 = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.side_14 = nn.Conv2d(3, 3, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_14 = nn.ZeroPad2d(padding=(1, 0, 0, 1))
        self.attendtion = nn.Conv2d(3 * 9, 3, kernel_size=1, stride=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.side_all_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.ZeroPad_all_1 = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.side_21_1 = nn.Conv2d(64, 64, kernel_size=(2, 3), stride=1, bias=False)
        self.ZeroPad_21_1 = nn.ZeroPad2d(padding=(1, 1, 1, 0))
        self.side_22_1 = nn.Conv2d(64, 64, kernel_size=(3, 2), stride=1, bias=False)
        self.ZeroPad_22_1 = nn.ZeroPad2d(padding=(0, 1, 1, 1))
        self.side_23_1 = nn.Conv2d(64, 64, kernel_size=(2, 3), stride=1, bias=False)
        self.ZeroPad_23_1 = nn.ZeroPad2d(padding=(1, 1, 0, 1))
        self.side_24_1 = nn.Conv2d(64, 64, kernel_size=(3, 2), stride=1, bias=False)
        self.ZeroPad_24_1 = nn.ZeroPad2d(padding=(1, 0, 1, 1))
        self.side_11_1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_11_1 = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.side_12_1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_12_1 = nn.ZeroPad2d(padding=(0, 1, 1, 0))
        self.side_13_1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_13_1 = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.side_14_1 = nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, bias=False)
        self.ZeroPad_14_1 = nn.ZeroPad2d(padding=(1, 0, 0, 1))
        self.attendtion_1 = nn.Conv2d(64 * 9, 64, kernel_size=1, stride=1, bias=False)

        self.act = nn.ReLU(inplace=True)

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
        self.fc = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_all = self.ZeroPad_all(x)
        x_all = self.side_all(x_all)
        x_21 = self.ZeroPad_21(x)
        x_21 = self.side_21(x_21)
        x_22 = self.ZeroPad_22(x)
        x_22 = self.side_22(x_22)
        x_23 = self.ZeroPad_23(x)
        x_23 = self.side_23(x_23)
        x_24 = self.ZeroPad_24(x)
        x_24 = self.side_24(x_24)
        x_11 = self.ZeroPad_21(x)
        x_11 = self.side_21(x_11)
        x_12 = self.ZeroPad_22(x)
        x_12 = self.side_22(x_12)
        x_13 = self.ZeroPad_23(x)
        x_13 = self.side_23(x_13)
        x_14 = self.ZeroPad_24(x)
        x_14 = self.side_24(x_14)
        out_merge = torch.cat([x_all, x_21, x_22, x_23, x_24, x_11, x_12, x_13, x_14], dim=1)
        x = self.attendtion(out_merge)

        x = self.conv1(x)
        x_all_1 = self.ZeroPad_all_1(x)
        x_all_1 = self.side_all_1(x_all_1)
        x_21_1 = self.ZeroPad_21_1(x)
        x_21_1 = self.side_21_1(x_21_1)
        x_22_1 = self.ZeroPad_22_1(x)
        x_22_1 = self.side_22_1(x_22_1)
        x_23_1 = self.ZeroPad_23_1(x)
        x_23_1 = self.side_23_1(x_23_1)
        x_24_1 = self.ZeroPad_24_1(x)
        x_24_1 = self.side_24_1(x_24_1)
        x_11_1 = self.ZeroPad_21_1(x)
        x_11_1 = self.side_21_1(x_11_1)
        x_12_1 = self.ZeroPad_22_1(x)
        x_12_1 = self.side_22_1(x_12_1)
        x_13_1 = self.ZeroPad_23_1(x)
        x_13_1 = self.side_23_1(x_13_1)
        x_14_1 = self.ZeroPad_24_1(x)
        x_14_1 = self.side_24_1(x_14_1)
        out_merge_1 = torch.cat([x_all_1, x_21_1, x_22_1, x_23_1, x_24_1, x_11_1, x_12_1, x_13_1, x_14_1], dim=1)
        x = self.attendtion_1(out_merge_1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)

        return x
