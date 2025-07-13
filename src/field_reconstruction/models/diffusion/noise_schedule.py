import torch
import torch.nn as nn
import math

def ddpm_schedules(beta1: float = 0.0001, beta2: float = 0.02, num_timesteps: int = 1000):
    """
    Generate a linear schedule for beta values used in DDPM.

    Args:
        beta1 (float): The starting value of beta.
        beta2 (float): The ending value of beta.
        num_timesteps (int): The number of timesteps.

    Returns:
        list: A list of beta values for each timestep.
    """
    
    assert beta1 < beta2, "beta1 must be less than beta2"
    
    beta_t = torch.linspace(beta1, beta2, num_timesteps + 1)
    
    # Compute alpha t and alpha bar t (cumulative product)
    alpha_t = 1.0 - beta_t  
    alphabar_t = torch.cumprod(alpha_t, dim=0)  
    
    # Compute other necessary terms
    oneover_sqrta = 1.0 / torch.sqrt(alpha_t)  
    sqrt_beta_t = torch.sqrt(beta_t) 
    sqrtab = torch.sqrt(alphabar_t) 
    sqrtmab = torch.sqrt(1 - alphabar_t) 
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab 
    #######################################################################
    #                       ** END OF YOUR CODE **
    ####################################################################### 

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
    
class ConvBlock(nn.Module):
    """
    A simple convolutional block with optional batch normalization and ReLU activation.
    """

    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.sampling = up

        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)  # 1st part
        self.batchnorm_1 = nn.BatchNorm2d(out_ch)

        self.linear = nn.Linear(time_emb_dim, out_ch)  # For time embeddings

        self.conv_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)  # 2nd part
        self.batchnorm_2 = nn.BatchNorm2d(out_ch)

        if self.sampling:
            # use transpose for up
            self.conv_3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)  # 3rd part
        else:
            self.conv_3 = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=1, padding=0)

        self.relu = nn.ReLU()
        
    def forward(self, x, t):
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
    
#TODO: Implement Unet, Positional Embedding, and Diffusion Model...
    