import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    
    def __init__(self, in_channels=2, out_channels=1, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(4,2), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(4,2), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec6 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoding: goes from (B, in_channels, 64, 32) to (B, 512, 1, 1)
        e1 = self.enc1(x)  
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2) 
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5) # -> (B, 512, 1, 1)

        # Decoding with skip connections: goes from (B, 512, 1, 1) to (B, out_channels, 64, 32)
        d1 = self.dec1(e6)        
        d2 = self.dec2(torch.cat([d1, e5], dim=1)) 
        d3 = self.dec3(torch.cat([d2, e4], dim=1)) 
        d4 = self.dec4(torch.cat([d3, e3], dim=1)) 
        d5 = self.dec5(torch.cat([d4, e2], dim=1)) 
        out = self.dec6(torch.cat([d5, e1], dim=1)) # -> (B, out_channels, 64, 32)
        return out
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=(4,2), stride=1, padding=0) #For 64x32 input
        
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        return x