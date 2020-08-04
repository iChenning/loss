import torch.nn as nn
import torch.nn.functional as F


class SideNet5(nn.Module):
    def __init__(self):
        super(SideNet5, self).__init__()
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
        self.attendtion = nn.Conv2d(15, 3, kernel_size=1, stride=1, bias=False)

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
        self.fc = nn.Sequential(
            nn.Linear(1024, 128, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_all = self.ZeroPad_all(x)
        x_all = self.side_all(x_all)
        x_21 = self.ZeroPad_all(x)
        x_21 = self.side_all(x_21)
        x_22 = self.ZeroPad_all(x)
        x_22 = self.side_all(x_22)
        x_23 = self.ZeroPad_all(x)
        x_23 = self.side_all(x_23)
        x_24 = self.ZeroPad_all(x)
        x_24 = self.side_all(x_24)
        out_merge = torch.cat([x_all, x_21, x_22, x_23, x_24], dim=1)
        x = self.attendtion(out_merge)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)

        return x
