import torch
import torch.nn as nn


class FukamiNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(FukamiNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, padding=3),  # same padding
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, out_channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class FukamiResNet(nn.Module):  # test
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=48, num_blocks=6):
        super(FukamiResNet, self).__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=7, padding=3, bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )
        self.exit = nn.Conv2d(hidden_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_blocks(x)
        return self.exit(x)


class FukamiUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64):
        super(FukamiUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        # Output
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Decoder
        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.final(x)
