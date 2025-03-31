import torch
import torch.nn as nn
import torch.nn.functional as F

class FukamiNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super(FukamiNet, self).__init__()

        layers = []
        current_channels = in_channels

        # Encoder
        for f in features:
            layers.append(nn.Conv2d(current_channels, f, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            current_channels = f

        # Decoder
        for f in reversed(features):
            layers.append(nn.ConvTranspose2d(current_channels, f, kernel_size=2, stride=2))
            layers.append(nn.ReLU(inplace=True))
            current_channels = f

        # Final output layer
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)