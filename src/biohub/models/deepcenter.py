import torch
from torch import nn

from biohub.modules.conv import ConvBlock3d


class DeepCenterUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        width = int(base_channels)
        self.enc1 = ConvBlock3d(in_channels, width)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(width, width * 2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3d(width * 2, width * 4)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3d(width * 4, width * 8)
        self.up3 = nn.ConvTranspose3d(width * 8, width * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3d(width * 8, width * 4)
        self.up2 = nn.ConvTranspose3d(width * 4, width * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(width * 4, width * 2)
        self.up1 = nn.ConvTranspose3d(width * 2, width, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(width * 2, width)
        self.head = nn.Conv3d(width, 1, kernel_size=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(volume)
        skip2 = self.enc2(self.down1(skip1))
        skip3 = self.enc3(self.down2(skip2))
        hidden = self.bottleneck(self.down3(skip3))
        decoded = self.dec3(torch.cat([self.up3(hidden), skip3], dim=1))
        decoded = self.dec2(torch.cat([self.up2(decoded), skip2], dim=1))
        decoded = self.dec1(torch.cat([self.up1(decoded), skip1], dim=1))
        return self.head(decoded)
