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
    def __init__(self, in_channels=2, out_channels=1, hidden_channels=256, num_blocks=6):
        """
        Also Called VT-Unet.
        """
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

class VTUnet(nn.Module):
    """
    Deterministic UNet with the same architecture as SimpleUnet but without time conditioning.
    Direct mapping: 5 VT fields + 6 sensor channels -> 5 reconstructed fields
    Trained with MSE loss for deterministic field reconstruction.
    """
    def __init__(self, in_channels=6, out_channels=5):
    # 5 VT + 6 sensor = 11 input
        super().__init__()
        print(f"Using deterministic UNet with {in_channels} input channels and {out_channels} output channels")
        
        # Same architecture as SimpleUnet
        down_channels = (128, 256, 512)
        up_channels = (512, 256, 128)
        out_dim = out_channels

        # Initial projection (same as SimpleUnet conv_0)
        self.conv_0 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsampling blocks (same as SimpleUnet but no time/conditioning)
        self.downsampling = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            block = DeterministicBlock(down_channels[i], down_channels[i + 1], up=False)
            self.downsampling.append(block)
            
        # Bottleneck (same as SimpleUnet but no time/conditioning)
        self.bottleneck = DeterministicBlock(down_channels[-1], down_channels[-1], up=False)

        # Upsampling blocks (same as SimpleUnet but no time/conditioning)
        self.upsampling = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            self.upsampling.append(DeterministicBlock(up_channels[i] + down_channels[-i-1], up_channels[i+1], up=True))
        
        # Output projection (same as SimpleUnet)
        self.output_proj = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass for deterministic reconstruction.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 11, height, width)
                             [5 VT fields + 6 sensor channels]
        
        Returns:
            torch.Tensor: Reconstructed fields of shape (batch_size, 5, height, width)
        """
        # Initial convolution
        x = self.conv_0(x)
        res = []

        # Downsampling path
        for down in self.downsampling:
            x, _ = down(x)
            res.append(x)

        # Bottleneck
        _, x = self.bottleneck(x)
        
        # Upsampling path with skip connections
        for i in range(len(self.upsampling)):
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1)
            x, _ = self.upsampling[i](x)

        x = self.output_proj(x)
        return x


class DeterministicBlock(nn.Module):
    """
    Block for deterministic UNet - same as Block but without time embeddings and conditioning.
    """
    
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.sampling = up
        
        # Same convolution structure as original Block
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_ch)  
        
        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_ch)

        # Same sampling operations as original Block
        if self.sampling:
            self.conv_3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass without time embeddings or conditioning.
        
        Args:
            x: Input features
            
        Returns:
            h: Features after processing (for skip connections)
            z: Sampled features (for next level)
        """
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        h = self.relu(x) 
        
        # No time embeddings or conditioning - just process features
        h = self.conv_2(h)
        h = self.batchnorm_2(h)
        h = self.relu(h)
        
        z = self.conv_3(h)
        
        return h, z
        
        
    