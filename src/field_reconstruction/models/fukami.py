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