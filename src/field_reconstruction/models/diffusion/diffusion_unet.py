import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Block(nn.Module):
    """
    A basic building block for the U-Net architecture that processes both spatial and temporal information.
    
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels 
        time_emb_dim (int): Dimension of time embedding
        up (bool): If True, uses transposed convolution for upsampling. If False, uses regular convolution for downsampling
    """
    
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.sampling = up
        
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding= 1) #1st part
        self.batchnorm_1 = nn.BatchNorm2d(out_ch)  
        
        self.linear = nn.Linear(time_emb_dim, out_ch) #For time embeddings

        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding = 1) #2nd part
        self.batchnorm_2 = nn.BatchNorm2d(out_ch)

        if self.sampling:
            # use transpose for up 
            self.conv_3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size = 2, stride = 2) #3rd part
        else:
            self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size = 2, stride = 2, padding = 0)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        Forward pass of the block.
        
        Args:
            x (torch.Tensor): Input feature maps
            t (torch.Tensor): Time embeddings
            
        Returns:
            torch.Tensor: Transformed feature maps
        """
        x = self.conv_1(x) #1st part
        x = self.batchnorm_1(x)
        h = self.relu(x) 
        
        t = self.linear(t) #Time embeddings
        t = t.unsqueeze(-1).unsqueeze(-1)
        t = t.expand(-1, -1, h.shape[2], h.shape[3])
        
        h = h + t #Add time to features
        
        h = self.conv_2(h) #2nd part
        h = self.batchnorm_2(h)
        h = self.relu(h)
        
        z = self.conv_3(h) #3rd part
        
        return h,z 
    
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Creates sinusoidal positional embeddings for time steps.
    Uses alternating sine and cosine functions at different frequencies.
    
    Args:
        dim (int): Dimension of the embeddings
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Compute positional embeddings for given timesteps.
        
        Args:
            time (torch.Tensor): Tensor of timesteps
            
        Returns:
            torch.Tensor: Position embeddings of shape (batch_size, dim)
        """
        time = time.unsqueeze(-1)  
        half_dimension = self.dim // 2 
        exponent = torch.arange(half_dimension, device=time.device).float() / half_dimension
        freq = 1e-4 ** exponent  
        
        thetas = time * freq 
        embeddings = torch.cat([thetas.sin(), thetas.cos()], dim=-1)  
        
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture for diffusion models.
    Includes time conditioning and skip connections.

    Args:
        in_channels (int): Number of input image channels
    """
    def __init__(self, in_channels=11):  # 5 previous step + 6 conditioning
        super().__init__()
        image_channels = in_channels
        print(f"Using {image_channels} input channels for the diffusion model.")
        down_channels = (128, 256, 512)  # Limited the downsampling stages
        up_channels = (512, 256, 128)
        out_dim = 5  # Output only the 5 denoised channels
        time_emb_dim = 256

        # Time embedding layers
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv_0 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3,  padding = 1) #Initial projection


        self.downsampling = nn.ModuleList() #Downsampling part
        for out_chn in range(len(down_channels)  - 1):
            block = Block(down_channels[out_chn], down_channels[out_chn + 1], time_emb_dim, up=False)
            self.downsampling.append(block)
            
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim, up=False) # Bottleneck bloc

        self.upsampling = nn.ModuleList() #Upsampling part
        for i in range(len(up_channels) - 1):
            self.upsampling.append(Block(up_channels[i] + down_channels[-i-1], up_channels[i+1], time_emb_dim, up=True))
        
        
        self.output_proj = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1) #Need to projeco

    def forward(self, x, timestep, cond):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 11, height, width)
            timestep (torch.Tensor): Current timestep for conditioning

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 5, height, width)
        """
        # Get time embeddings
        t = self.time_mlp(timestep) 
        
        #Concatenate x with conditioning
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        
        if x.shape[1] != 11:  # Ensure input has 11 channels
            raise ValueError(f"Input tensor must have 11 channels, got {x.shape[1]} channels instead.")
        # Initial convolution
        x = self.conv_0(x)
        
        # Store intermediate outputs for skip connections
        res = []

        # Downsampling path
        for down in self.downsampling:
            x, _ = down(x, t)  
            res.append(x)  

        # Bottleneck
        _, x = self.bottleneck(x, t) 
        
        # Upsampling path with skip connections
        for i in range(len(self.upsampling)): #Upsampling
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1) #Add the skip-connexions
            
            x, _ = self.upsampling[i](x, t)

        x = self.output_proj(x)

        return x  
        