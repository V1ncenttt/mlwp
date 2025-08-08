import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from abc import ABC, abstractmethod

class ConditioningMethod(ABC):
    """Abstract base class for conditioning methods."""
    
    @abstractmethod
    def __init__(self, cond_channels, feature_channels, **kwargs):
        pass
    
    @abstractmethod
    def encode_conditioning(self, cond):
        """Encode raw conditioning into the format needed by this method."""
        pass
    
    @abstractmethod
    def apply_conditioning(self, features, cond_encoded, feature_size):
        """Apply conditioning to features at a specific resolution level."""
        pass


class SpatialConditioning(ConditioningMethod):
    """Original spatial conditioning method (our current approach)."""
    
    def __init__(self, cond_channels, feature_channels, **kwargs):
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cond_channels, 256, 1),  # Point-wise conv
        )
        self.cond_conv = nn.Conv2d(256, feature_channels, 1)
    
    def encode_conditioning(self, cond):
        """Returns spatial conditioning maps."""
        return self.cond_encoder(cond)
    
    def apply_conditioning(self, features, cond_encoded, feature_size):
        """Apply spatial conditioning via addition."""
        # Resize conditioning to match feature map size
        cond_resized = F.interpolate(cond_encoded, size=feature_size, mode='bilinear')
        # Project to match feature channels
        cond_projected = self.cond_conv(cond_resized)
        return features + cond_projected

class FiLMConditioning(ConditioningMethod):
    """Feature-wise Linear Modulation (FiLM) conditioning."""
    
    def __init__(self, cond_channels, feature_channels, **kwargs):
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global pooling for FiLM
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # FiLM parameters: scale (gamma) and shift (beta)
        self.gamma_layer = nn.Linear(128, feature_channels)
        self.beta_layer = nn.Linear(128, feature_channels)
    
    def encode_conditioning(self, cond):
        """Returns global conditioning vector."""
        return self.cond_encoder(cond)
    
    def apply_conditioning(self, features, cond_encoded, feature_size):
        """Apply FiLM modulation: gamma * features + beta."""
        # Generate scale and shift parameters
        gamma = self.gamma_layer(cond_encoded).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_layer(cond_encoded).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        
        # Apply FiLM modulation
        return gamma * features + beta

  
class Block(nn.Module):
    """
    A basic building block for the U-Net architecture that processes spatial, temporal, and conditioning information.
    
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels 
        time_emb_dim (int): Dimension of time embedding
        cond_emb_dim (int, optional): Dimension of conditioning embedding
        up (bool): If True, uses transposed convolution for upsampling. If False, uses regular convolution for downsampling
    """
    
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim=None, up=False):
        super().__init__()
        self.sampling = up
        self.has_conditioning = cond_emb_dim is not None
        
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) #1st part
        self.batchnorm_1 = nn.BatchNorm2d(out_ch)  
        
        # Time embedding projection
        self.time_linear = nn.Linear(time_emb_dim, out_ch) #For time embeddings
        
        # Conditioning embedding projection
        if self.has_conditioning:
            # For spatial conditioning - project channels to match output channels
            self.cond_conv = nn.Conv2d(cond_emb_dim, out_ch, kernel_size=1)

        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) #2nd part
        self.batchnorm_2 = nn.BatchNorm2d(out_ch)

        if self.sampling:
            # use transpose for up 
            self.conv_3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2) #3rd part
        else:
            self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x, t, cond_emb=None):
        """
        Forward pass of the block with time and conditioning embeddings.
        
        Args:
            x (torch.Tensor): Input feature maps
            t (torch.Tensor): Time embeddings
            cond_emb (torch.Tensor, optional): Conditioning embeddings
            
        Returns:
            tuple: (intermediate_features, output_features)
        """
        x = self.conv_1(x) #1st part
        x = self.batchnorm_1(x)
        h = self.relu(x) 
        
        # Add time embeddings
        t_emb = self.time_linear(t) #Time embeddings
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, h.shape[2], h.shape[3])
        h = h + t_emb #Add time to features
        
        # Add conditioning embeddings if available
        if self.has_conditioning and cond_emb is not None:
            # Resize spatial conditioning to match current feature map size
            cond_resized = F.interpolate(cond_emb, size=(h.shape[2], h.shape[3]), mode='bilinear', align_corners=False)
            # Project conditioning channels to match feature channels
            c_emb = self.cond_conv(cond_resized)
            h = h + c_emb  # Add spatial conditioning to features
        
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
    A simplified variant of the Unet architecture for diffusion models with deep conditioning.
    Includes time conditioning, deep conditioning, and skip connections.

    Args:
        in_channels (int): Number of input image channels (only for the target data)
        cond_channels (int): Number of conditioning channels
    """
    def __init__(self, in_channels=5, cond_channels=6):
        super().__init__()
        print(f"Using {in_channels} input channels and {cond_channels} conditioning channels for the diffusion model.")
        
        down_channels = (128, 256, 512)  # Limited the downsampling stages
        up_channels = (512, 256, 128)
        out_dim = 5  # Output only the 5 denoised channels
        time_emb_dim = 256
        cond_emb_dim = 128  # Conditioning embedding dimension

        # Time embedding layers
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Conditioning encoder - preserves spatial information
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, cond_emb_dim, kernel_size=1),  # Point-wise conv to get desired channels
            # Remove AdaptiveAvgPool2d - preserve spatial dimensions!
        )
        
        # Initial projection (only for target data, no conditioning concatenation)
        self.conv_0 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1) #Initial projection

        # Downsampling with conditioning support
        self.downsampling = nn.ModuleList() #Downsampling part
        for out_chn in range(len(down_channels) - 1):
            block = Block(down_channels[out_chn], down_channels[out_chn + 1], time_emb_dim, cond_emb_dim, up=False)
            self.downsampling.append(block)
            
        # Bottleneck with conditioning
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim, cond_emb_dim, up=False) # Bottleneck bloc

        # Upsampling with conditioning support
        self.upsampling = nn.ModuleList() #Upsampling part
        for i in range(len(up_channels) - 1):
            self.upsampling.append(Block(up_channels[i] + down_channels[-i-1], up_channels[i+1], time_emb_dim, cond_emb_dim, up=True))
        
        self.output_proj = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1) #Need to projeco

    def forward(self, x, timestep, cond):
        """
        Forward pass of the U-Net with deep conditioning.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5, height, width) - target data only
            timestep (torch.Tensor): Current timestep for conditioning
            cond (torch.Tensor): Conditioning tensor of shape (batch_size, 6, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 5, height, width)
        """
        # Get time embeddings
        t = self.time_mlp(timestep) 
        
        # Encode conditioning to embedding vector
        cond_emb = None
        if cond is not None:
            cond_emb = self.cond_encoder(cond)
        
        # Ensure input has correct number of channels (target data only)
        if x.shape[1] != 5:
            raise ValueError(f"Input tensor must have 5 channels, got {x.shape[1]} channels instead.")
        
        # Initial convolution (no concatenation - conditioning is handled via embeddings)
        x = self.conv_0(x)
        
        # Store intermediate outputs for skip connections
        res = []

        # Downsampling path with conditioning at every level
        for down in self.downsampling:
            x, _ = down(x, t, cond_emb)  # Pass conditioning to every block
            res.append(x)  

        # Bottleneck with conditioning
        _, x = self.bottleneck(x, t, cond_emb) 
        
        # Upsampling path with skip connections and conditioning
        for i in range(len(self.upsampling)): #Upsampling
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1) #Add the skip-connexions
            
            # Apply upsampling block with conditioning
            x, _ = self.upsampling[i](x, t, cond_emb)

        x = self.output_proj(x)

        return x  
    
class UnconditionalUnet(nn.Module):
    """
    A simplified variant of the Unet architecture for diffusion models.
    Includes time conditioning and skip connections.

    Args:
        in_channels (int): Number of input image channels
    """
    def __init__(self, in_channels=5):  # 5 previous step + 6 conditioning
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
            block = Block(down_channels[out_chn], down_channels[out_chn + 1], time_emb_dim, cond_emb_dim=None, up=False)
            self.downsampling.append(block)
            
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim, cond_emb_dim=None, up=False) # Bottleneck bloc

        self.upsampling = nn.ModuleList() #Upsampling part
        for i in range(len(up_channels) - 1):
            self.upsampling.append(Block(up_channels[i] + down_channels[-i-1], up_channels[i+1], time_emb_dim, cond_emb_dim=None, up=True))
        
        
        self.output_proj = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1) #Need to projeco

    def forward(self, x, timestep):
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
        

        if x.shape[1] != 5:  # Ensure input has 5 channels
            raise ValueError(f"Input tensor must have 5 channels, got {x.shape[1]} channels instead.")
        # Initial convolution
        x = self.conv_0(x)
        
        # Store intermediate outputs for skip connections
        res = []

        # Downsampling path
        for down in self.downsampling:
            x, _ = down(x, t, cond_emb=None)  
            res.append(x)  

        # Bottleneck
        _, x = self.bottleneck(x, t, cond_emb=None) 
        
        # Upsampling path with skip connections
        for i in range(len(self.upsampling)): #Upsampling
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1) #Add the skip-connexions
            
            x, _ = self.upsampling[i](x, t, cond_emb=None)

        x = self.output_proj(x)

        return x
        