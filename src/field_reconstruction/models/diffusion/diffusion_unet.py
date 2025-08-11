import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from abc import ABC, abstractmethod

class ConditioningMethod(nn.Module, ABC):
    """Abstract base class for conditioning methods."""
    
    def __init__(self, cond_channels, feature_channels, **kwargs):
        super().__init__()
    
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
        super().__init__(cond_channels, feature_channels, **kwargs)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1),  # Point-wise conv - fixed from cond_channels to 256
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
        super().__init__(cond_channels, feature_channels, **kwargs)
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
    def __init__(self, in_ch, out_ch, time_emb_dim, conditioning_method=None, up=False):
        super().__init__()
        self.sampling = up
        self.conditioning_method = conditioning_method
        
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_ch)  
        
        # Time embedding projection
        self.time_linear = nn.Linear(time_emb_dim, out_ch)

        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_ch)

        if self.sampling:
            self.conv_3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x, t, cond_encoded=None):
        """
        Forward pass with pluggable conditioning.
        
        Args:
            x: Input features
            t: Time embeddings
            cond_encoded: Pre-encoded conditioning (format depends on conditioning method)
        """
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        h = self.relu(x) 
        
        # Add time embeddings
        t_emb = self.time_linear(t)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, h.shape[2], h.shape[3])
        h = h + t_emb
        
        # Apply conditioning if available
        if self.conditioning_method is not None and cond_encoded is not None:
            h = self.conditioning_method.apply_conditioning(h, cond_encoded, (h.shape[2], h.shape[3]))
        
        h = self.conv_2(h)
        h = self.batchnorm_2(h)
        h = self.relu(h)
        
        z = self.conv_3(h)
        
        return h, z
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
    def __init__(self, in_channels=5, cond_channels=6, conditioning_type="spatial", **conditioning_kwargs):
        super().__init__()
        print(f"Using {conditioning_type} conditioning with {in_channels} input channels and {cond_channels} conditioning channels")
        
        down_channels = (128, 256, 512)
        up_channels = (512, 256, 128)
        out_dim = 5
        time_emb_dim = 256

        # Time embedding layers
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Create conditioning method based on type
        self.conditioning_type = conditioning_type
        conditioning_methods = {
            "spatial": SpatialConditioning,
            "film": FiLMConditioning,
        }
        
        if conditioning_type not in conditioning_methods:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")
        
        # Create conditioning instances for each resolution level with correct channel matching
        self.conditioning_methods = nn.ModuleDict()
        
        # Downsampling blocks: condition on output channels of each block
        for i in range(len(down_channels) - 1):
            out_channels = down_channels[i + 1]  # Output channels of this downsampling block
            self.conditioning_methods[f"down_{i}"] = conditioning_methods[conditioning_type](
                cond_channels, out_channels, **conditioning_kwargs
            )
            
        # Bottleneck: condition on bottleneck channels
        self.conditioning_methods["bottleneck"] = conditioning_methods[conditioning_type](
            cond_channels, down_channels[-1], **conditioning_kwargs
        )
        
        # Upsampling blocks: condition on output channels of each block
        for i in range(len(up_channels) - 1):
            out_channels = up_channels[i + 1]  # Output channels of this upsampling block
            self.conditioning_methods[f"up_{i}"] = conditioning_methods[conditioning_type](
                cond_channels, out_channels, **conditioning_kwargs
            )
        
        # Initial projection
        self.conv_0 = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        self.downsampling = nn.ModuleList()
        for i in range(len(down_channels) - 1):
            conditioning_method = self.conditioning_methods[f"down_{i}"]
            block = Block(down_channels[i], down_channels[i + 1], time_emb_dim, conditioning_method, up=False)
            self.downsampling.append(block)
            
        # Bottleneck
        bottleneck_conditioning = self.conditioning_methods["bottleneck"]
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim, bottleneck_conditioning, up=False)

        # Upsampling blocks
        self.upsampling = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            conditioning_method = self.conditioning_methods[f"up_{i}"]
            self.upsampling.append(Block(up_channels[i] + down_channels[-i-1], up_channels[i+1], time_emb_dim, conditioning_method, up=True))
        
        self.output_proj = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x, timestep, cond):
        """Forward pass with conditioning type specified at init."""
        # Get time embeddings
        t = self.time_mlp(timestep)
        
        # Encode conditioning once (format depends on conditioning type)
        cond_encoded_levels = {}
        if cond is not None:
            for level_name, conditioning_method in self.conditioning_methods.items():
                cond_encoded_levels[level_name] = conditioning_method.encode_conditioning(cond)
        
        # Initial convolution
        x = self.conv_0(x)
        res = []

        # Downsampling with level-specific conditioning
        for i, down in enumerate(self.downsampling):
            level_cond = cond_encoded_levels.get(f"down_{i}")
            x, _ = down(x, t, level_cond)
            res.append(x)

        # Bottleneck
        bottleneck_cond = cond_encoded_levels.get("bottleneck")
        _, x = self.bottleneck(x, t, bottleneck_cond)
        
        # Upsampling
        for i in range(len(self.upsampling)):
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1)
            
            level_cond = cond_encoded_levels.get(f"up_{i}")
            x, _ = self.upsampling[i](x, t, level_cond)

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
            block = Block(down_channels[out_chn], down_channels[out_chn + 1], time_emb_dim, conditioning_method=None, up=False)
            self.downsampling.append(block)
            
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim, conditioning_method=None, up=False) # Bottleneck bloc

        self.upsampling = nn.ModuleList() #Upsampling part
        for i in range(len(up_channels) - 1):
            self.upsampling.append(Block(up_channels[i] + down_channels[-i-1], up_channels[i+1], time_emb_dim, conditioning_method=None, up=True))
        
        
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
            x, _ = down(x, t, cond_encoded=None)  
            res.append(x)  

        # Bottleneck
        _, x = self.bottleneck(x, t, cond_encoded=None) 
        
        # Upsampling path with skip connections
        for i in range(len(self.upsampling)): #Upsampling
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1) #Add the skip-connexions
            
            x, _ = self.upsampling[i](x, t, cond_encoded=None)

        x = self.output_proj(x)

        return x
        