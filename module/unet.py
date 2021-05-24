import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels: int):
        super(UNet, self).__init__()
        self.in_channels = in_channels

        self.inc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.down_00 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.down_01 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(self.in_channels * 2, self.in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels * 2),
            nn.ReLU()
        )

        self.down_02 = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(self.in_channels * 2, self.in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels * 2),
            nn.ReLU()
        )

        self.up_00 = nn.Sequential(
            nn.Conv2d(self.in_channels * 4, self.in_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.up_01 = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.up_02 = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.outc = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1)

        self.inc_to_down00 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.down00_to_down01 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        self.down01_to_down02 = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.in_channels * 2),
            nn.ReLU()
        )

        self.upconv_00_to_01 = nn.ConvTranspose2d(in_channels=self.in_channels * 2, out_channels=self.in_channels * 2, kernel_size=4, stride=2, padding=1)
        self.upconv_01_to_02 = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=4, stride=2, padding=1)
        self.upconv_02_to_out = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):

        inc_out = self.inc(x)
        down_00_out = self.down_00(self.inc_to_down00(inc_out))
        down_01_out = self.down_01(self.down00_to_down01(down_00_out))
        down_02_out = self.down_02(self.down01_to_down02(down_01_out))

        up_00_out = self.up_00(torch.cat([nn.Upsample(size=tuple(down_01_out.shape[2:]))(self.upconv_00_to_01(down_02_out)), down_01_out], dim=1))
        up_01_out = self.up_01(torch.cat([nn.Upsample(size=tuple(down_00_out.shape[2:]))(self.upconv_01_to_02(up_00_out)), down_00_out], dim=1))
        up_02_out = self.up_02(torch.cat([nn.Upsample(size=tuple(inc_out.shape[2:]))(self.upconv_02_to_out(up_01_out)), inc_out], dim=1))

        # up_00_out = self.up_00(torch.cat([self.upconv_00_to_01(down_02_out), down_01_out], dim=1))
        # up_01_out = self.up_01(torch.cat([self.upconv_01_to_02(up_00_out), down_00_out], dim=1))
        # up_02_out = self.up_02(torch.cat([self.upconv_02_to_out(up_01_out), inc_out], dim=1))

        out = self.outc(up_02_out)

        return out