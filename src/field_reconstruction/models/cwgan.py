import torch
import torch.nn as nn
import torch.nn.functional as F
class Generator(nn.Module):
    """
    U-Net Generator for image-to-image translation.
    Encodes the input to a bottleneck representation and decodes back with skip connections.
    Adapted for 64x32 input to support 6 encoder steps.
    """
    
    def __init__(self, in_channels=2, out_channels=1, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        # Encoder blocks
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # 64x32 -> 32x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x16 -> 16x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x8 -> 8x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x4 -> 4x2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=(0,0)),  # 4x2 -> 1x1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.encoder6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),  # 1x1 -> 1x1 (bottleneck conv)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder blocks
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=1, stride=1, padding=0),  # 1x1 -> 1x1
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=(0,0)),  # 1x1 -> 4x2
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 4x2 -> 8x4
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # 8x4 -> 16x8
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 16x8 -> 32x16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 32x16 -> 64x32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=1, padding=1)  # 64x32 -> 64x32

        initialize_weights_normal(self)

    def forward(self, x):
        # Encoder pathway, save outputs for skip connections
        enc1_out = self.encoder1(x)  
        enc2_out = self.encoder2(enc1_out) 
        enc3_out = self.encoder3(enc2_out) 
        enc4_out = self.encoder4(enc3_out)
        enc5_out = self.encoder5(enc4_out)
        bottleneck = self.encoder6(enc5_out)  # -> (B, 512, 1, 1)

        # Decoder pathway with skip connections concatenated
        dec1_out = self.decoder1(bottleneck)        
        dec2_out = self.decoder2(torch.cat([dec1_out, enc5_out], dim=1)) 
        dec3_out = self.decoder3(torch.cat([dec2_out, enc4_out], dim=1)) 
        dec4_out = self.decoder4(torch.cat([dec3_out, enc3_out], dim=1)) 
        dec5_out = self.decoder5(torch.cat([dec4_out, enc2_out], dim=1)) 
        dec6_out = self.decoder6(torch.cat([dec5_out, enc1_out], dim=1))
        out = self.final(torch.cat([dec6_out, x], dim=1))  # Adding input x as skip connection for stability
        return out
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0) #For 64x32 input

        initialize_weights_normal(self)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x

# Example of applying weight initialization:
# generator = Generator()
# discriminator = Discriminator()
# initialize_weights_normal(generator)
# initialize_weights_normal(discriminator)


def initialize_weights_normal(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)