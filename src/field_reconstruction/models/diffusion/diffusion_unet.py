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

class HybridConditioning(ConditioningMethod):
    "A conditioning method that combines FiLM and cross-attention. Based on Zhuang et al. 2024."
    
    def __init__(self, cond_channels, feature_channels, **kwargs):
        super().__init__(cond_channels, feature_channels, **kwargs)
        
        self.patch_size = kwargs.get('patch_size', 8)
        self.n_emb = kwargs.get('n_emb', 512)
        
        # Patchifiy with strided conv:
        self.patchify = nn.Conv2d(
            cond_channels,
            self.n_emb,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding=0
        )

        # FiLM layer
        self.film_layer = nn.Sequential(
            nn.Linear(self.n_emb, self.n_emb * 2),
            nn.ReLU(),
            nn.Linear(self.n_emb * 2, self.n_emb * 2),
        )
        
        # Pos Embedding
        self.pos_embedding = SinusoidalPatchEmbedding(self.n_emb)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.n_emb, num_heads=8, mlp_ratio=4)
            for _ in range(1)
        ])

        self.cross_attention = CrossAttentionLayer(feature_channels, self.n_emb)

    def encode_conditioning(self, cond):
        B, C, H, W = cond.shape
        
        patches = self.patchify(cond)
        patch_h, patch_w = patches.shape[2], patches.shape[3]
        n_patches = patch_h * patch_w
        
        patches_seq = patches.view(B, self.n_emb, n_patches).transpose(1, 2)
        
        film_params = self.film_layer(patches_seq)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        patches_film = gamma * patches_seq + beta
        
        pos_emb = self.pos_embedding(patch_h, patch_w, cond.device)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # Broadcast to batch size
        
        patches_with_pos = patches_film + pos_emb
        
        x = patches_with_pos
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        return {"patches": x, "patch_shape": (patch_h, patch_w), "n_patches": n_patches}
    
    def apply_conditioning(self, features, cond_encoded, feature_size):
        B, C, H, W = features.shape
        
        features_seq = features.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
        
        attended_features = self.cross_attention(features_seq, cond_encoded['patches'])
        
        conditioned_features = attended_features.transpose(1, 2).view(B, C, H, W)

        return conditioned_features

class SinusoidalPatchEmbedding(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, patch_h, patch_w, device):
        y_pos = torch.arange(patch_h, device=device, dtype=torch.float32)
        x_pos = torch.arange(patch_w, device=device, dtype=torch.float32)
        
        half_dim = self.dim // 4
        
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -(math.log(10000.0) / half_dim)
                          )
        y_emb = y_pos.unsqueeze(1) * freqs.unsqueeze(0)
        y_emb = torch.cat([y_emb.sin(), y_emb.cos()], dim=1)

        x_emb = x_pos.unsqueeze(1) * freqs.unsqueeze(0)
        x_emb = torch.cat([x_emb.sin(), x_emb.cos()], dim=1)

        pos_emb = torch.zeros(patch_h, patch_w, self.dim, device=device, dtype=torch.float32)
        pos_emb[:, :, :half_dim * 2] = y_emb.unsqueeze(1).expand(-1, patch_w, -1)
        pos_emb[:, :, half_dim * 2 : half_dim * 4] = x_emb.unsqueeze(0).expand(patch_h, -1, -1)
        
        return pos_emb.view(-1, self.dim)
    
class TransformerBlock(nn.Module):
    """
    Transformer block without LayerNorm, as shown in the diagram.
    Just Multi-Head Self-Attention + MLP with skip connections.
    """
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.3):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-Head Self-Attention (no LayerNorm as per diagram)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        # MLP (no LayerNorm as per diagram)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections as shown in diagram.
        
        Args:
            x: (B, n_patches, embed_dim) = (B, 32, 256)
            
        Returns:
            x: (B, n_patches, embed_dim) = (B, 32, 256)
        """
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x    

# Add this class after your TransformerBlock class:

class CrossAttentionLayer(nn.Module):
    """Cross-attention for applying patch conditioning to U-Net features."""
    
    def __init__(self, feature_dim, patch_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Project patch embeddings to feature dimension if needed
        self.patch_proj = nn.Linear(patch_dim, feature_dim) if patch_dim != feature_dim else nn.Identity()
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm (optional, but helps with training stability)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features, patches):
        """
        Args:
            features: (B, H*W, feature_dim) - flattened U-Net features
            patches: (B, n_patches, patch_dim) - processed conditioning patches (32 patches with 256-dim)
            
        Returns:
            attended_features: (B, H*W, feature_dim)
        """
        # Project patches to feature dimension if needed
        patches_proj = self.patch_proj(patches)  # (B, 32, feature_dim)
        
        # Cross-attention: features attend to patches
        attended, attn_weights = self.cross_attn(
            query=features,     # (B, H*W, feature_dim) - U-Net features asking "what conditioning do I need?"
            key=patches_proj,   # (B, 32, feature_dim) - conditioning patches as keys
            value=patches_proj,  # (B, 32, feature_dim) - conditioning patches as values
            average_attn_weights=False
        )
        
        # Skip connection and normalization

        return self.norm(features + attended)

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
            "hybrid": HybridConditioning,
        }
        
        if conditioning_type not in conditioning_methods:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")
        print(f"Using {conditioning_type} conditioning method.")
        
        # For hybrid conditioning, use shared encoding + level-specific projections
        if conditioning_type == "hybrid":
            # Create one shared conditioning encoder for all levels
            self.shared_conditioning = HybridConditioning(cond_channels, 512, **conditioning_kwargs)  # Use 512 as intermediate dim
            
            # Create level-specific cross-attention projections only
            self.conditioning_projections = nn.ModuleDict()
            
            # Downsampling blocks: project to output channels of each block
            for i in range(len(down_channels) - 1):
                out_channels = down_channels[i + 1]
                self.conditioning_projections[f"down_{i}"] = CrossAttentionLayer(out_channels, 512)
                
            # Bottleneck: project to bottleneck channels
            self.conditioning_projections["bottleneck"] = CrossAttentionLayer(down_channels[-1], 512)
            
            # Upsampling blocks: project to output channels of each block
            for i in range(len(up_channels) - 1):
                out_channels = up_channels[i + 1]
                self.conditioning_projections[f"up_{i}"] = CrossAttentionLayer(out_channels, 512)
        else:
            # For spatial and film conditioning, keep the original approach (separate instances per level)
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
            # For hybrid conditioning, pass None as conditioning_method since we handle it separately
            conditioning_method = None if conditioning_type == "hybrid" else self.conditioning_methods[f"down_{i}"]
            block = Block(down_channels[i], down_channels[i + 1], time_emb_dim, conditioning_method, up=False)
            self.downsampling.append(block)
            
        # Bottleneck
        bottleneck_conditioning = None if conditioning_type == "hybrid" else self.conditioning_methods["bottleneck"]
        self.bottleneck = Block(down_channels[-1], down_channels[-1], time_emb_dim, bottleneck_conditioning, up=False)

        # Upsampling blocks
        self.upsampling = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            conditioning_method = None if conditioning_type == "hybrid" else self.conditioning_methods[f"up_{i}"]
            self.upsampling.append(Block(up_channels[i] + down_channels[-i-1], up_channels[i+1], time_emb_dim, conditioning_method, up=True))
        
        self.output_proj = nn.Conv2d(up_channels[-1], out_dim, kernel_size=3, padding=1)

    def forward(self, x, timestep, cond):
        """Forward pass with conditioning type specified at init."""
        # Get time embeddings
        t = self.time_mlp(timestep)
        
        # Handle conditioning based on type
        if self.conditioning_type == "hybrid":
            # For hybrid conditioning: encode once, then apply level-specific projections
            if cond is not None:
                shared_cond_encoded = self.shared_conditioning.encode_conditioning(cond)
                shared_patches = shared_cond_encoded['patches']  # (B, n_patches, 512)
            else:
                # CFG: No conditioning - create zero patches for unconditional generation
                B = x.shape[0]
                device = x.device
                # Create zero conditioning patches with same shape as normal patches
                shared_patches = torch.zeros(B, 64, 512, device=device)  # (B, n_patches=64, embed_dim=512)
        else:
            # For spatial/film conditioning: encode separately for each level (original approach)
            cond_encoded_levels = {}
            if cond is not None:
                for level_name, conditioning_method in self.conditioning_methods.items():
                    cond_encoded_levels[level_name] = conditioning_method.encode_conditioning(cond)
            else:
                # CFG: Set all conditioning to None
                for level_name in self.conditioning_methods.keys():
                    cond_encoded_levels[level_name] = None
        
        # Initial convolution
        x = self.conv_0(x)
        res = []

        # Downsampling with level-specific conditioning
        for i, down in enumerate(self.downsampling):
            if self.conditioning_type == "hybrid":
                # Apply shared conditioning via cross-attention projection
                x, _ = down(x, t, None)  # No conditioning in Block itself
                # Apply hybrid conditioning manually after the block
                B, C, H, W = x.shape
                x_flat = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
                x_conditioned = self.conditioning_projections[f"down_{i}"](x_flat, shared_patches)
                x = x_conditioned.transpose(1, 2).view(B, C, H, W)  # Back to (B, C, H, W)
            else:
                level_cond = cond_encoded_levels.get(f"down_{i}")
                x, _ = down(x, t, level_cond)
            res.append(x)

        # Bottleneck
        if self.conditioning_type == "hybrid":
            _, x = self.bottleneck(x, t, None)  # No conditioning in Block itself
            # Apply hybrid conditioning manually after the block
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
            x_conditioned = self.conditioning_projections["bottleneck"](x_flat, shared_patches)
            x = x_conditioned.transpose(1, 2).view(B, C, H, W)  # Back to (B, C, H, W)
        else:
            bottleneck_cond = cond_encoded_levels.get("bottleneck")
            _, x = self.bottleneck(x, t, bottleneck_cond)
        
        # Upsampling
        for i in range(len(self.upsampling)):
            if x.shape[-2:] != res[-i-1].shape[-2:]:
                x = F.interpolate(x, size=res[-i-1].shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, res[-i-1]], dim=1)
            
            if self.conditioning_type == "hybrid":
                x, _ = self.upsampling[i](x, t, None)  # No conditioning in Block itself
                # Apply hybrid conditioning manually after the block
                B, C, H, W = x.shape
                x_flat = x.view(B, C, H*W).transpose(1, 2)  # (B, H*W, C)
                x_conditioned = self.conditioning_projections[f"up_{i}"](x_flat, shared_patches)
                x = x_conditioned.transpose(1, 2).view(B, C, H, W)  # Back to (B, C, H, W)
            else:
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
        