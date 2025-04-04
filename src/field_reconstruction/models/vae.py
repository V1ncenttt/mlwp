import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(ConvolutionalBlock, self).__init__()
        
        self.expand_ratio = expand_ratio
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
    
class ReconstructionVAE(nn.Module):
    
    def __init__(self, channels, latent_dim):
        super(ReconstructionVAE, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        # 64x32 input
        self.encoder = nn.Sequential(
            self.make_layer_encode(ConvolutionalBlock, in_channel=channels, out_channel=32, stride=2, num_blocks=1, expand_ratio=1),  # 64x32 → 32x16
            self.make_layer_encode(ConvolutionalBlock, in_channel=32, out_channel=64, stride=2, num_blocks=1, expand_ratio=1),         # 32x16 → 16x8
            nn.Flatten()
        )

        self.flat_dim = 64 * 16 * 8

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.decoder_linear = nn.Linear(latent_dim, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # Output only 1 channel
        )

    
    def make_layer_encode(self, block, in_channel, out_channel, stride, num_blocks, expand_ratio):
        layers = []
        for i in range(num_blocks):
            if i == 0: # Change this to be 0 for first layer 
                s = stride
            else:
                s = 1
            layers.append(block(in_channel, out_channel, s, expand_ratio))
            in_channel = out_channel
        return nn.Sequential(*layers)   
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) 

        mu = self.fc_mu(x) 
        
        logvar = self.fc_logvar(x) #Might add clamping
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent extreme logvar values

        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) #sample from normal
        z = mu + eps * std
        
        return z

    def decode(self, z):
        z = self.decoder_linear(z) 
        z = self.decoder(z) 
        return z
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1) 

        mu = self.fc_mu(x) 
        
        logvar = self.fc_logvar(x) #Might add clamping
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent extreme logvar values

        return mu, logvar
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar