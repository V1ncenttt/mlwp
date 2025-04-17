import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(48, out_channels, kernel_size=7, padding=3)
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

class FukamiResNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=48, num_blocks=6):
        super(FukamiResNet, self).__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.exit = nn.Conv2d(hidden_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.entry(x)
        x = self.res_blocks(x)
        return self.exit(x)

class FukamiUNet(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(FukamiUNet, self).__init__(*args, **kwargs)
    
    def forward(self, x):
        # Implement the forward pass for the UNet architecture
        pass