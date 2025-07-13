import torch
import torch.distributions as dist
from models.diffusion.noise_schedule import ddpm_schedules
import torch.nn as nn
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) implementation.
    
    This class implements the DDPM as described in "Denoising Diffusion Probabilistic Models" 
    (Ho et al., 2020). The model learns to reverse a gradual noising process.
    
    Args:
        eps_model (nn.Module): The neural network that predicts noise at each timestep
        betas (Tuple[float, float]): Beta schedule parameters (β_start, β_end)
        n_T (int): Number of timesteps in the diffusion process
        criterion (nn.Module): Loss function for training, defaults to MSELoss
    """
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss()
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v.to(device))  # move to device 

        self.n_T = n_T
        self.criterion = criterion

        # Initialize with dataset statistics for potentially better starting point
        mean =  0.04640255868434906 #TODO: Replace with dataset mean
        std_dev = 0.8382343649864197
        self.normal_dist = dist.Normal(mean, std_dev)
        

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Performs forward diffusion and predicts the noise added at timestep t.
        
        This implements Algorithm 1 from the DDPM paper:
        1. Sample a random timestep t
        2. Sample random noise ε
        3. Create noised input x_t using the noise schedule
        4. Predict the noise using the model
        5. Return the loss between predicted and actual noise
        
        Args:
            x (torch.Tensor): Input images/data
            t (torch.Tensor, optional): Specific timestep. If None, randomly sampled.
            
        Returns:
            Tuple containing:
            - Loss between predicted and actual noise
            - The sampled noise (eps)
            - The noised input at timestep t (x_t)
        """
         
        # Step 1: Sample timestep t if not provided
        if t is None:
            t = torch.randint(0, self.n_T, (x.shape[0],)).to(device)
            
        # Step 2: Sample noise from normal distribution
        epsilon = self.normal_dist.sample(x.shape).to(device) 
        
        # Step 3: Create noised input x_t using the forward process
        # x_t = √(αbar_t) * x_0 + √(1-αbar_t) * ε
        sqrtab_t = self.sqrtab[t].view(-1, 1, 1, 1).to(device)
        sqrtmab_t = self.sqrtmab[t].view(-1, 1, 1, 1).to(device)
        
        x_t = sqrtab_t * x + sqrtmab_t * epsilon

        # Step 4 & 5: Predict noise using eps_model and return loss
        # Note: timesteps are normalized to [0,1] range for the model
        predicted_epsilon = self.eps_model(x_t, t / self.n_T, cond)
        loss = self.criterion(predicted_epsilon, epsilon)
        
        return loss, epsilon, x_t
    
    def sample(self, n_sample: int, size, device, cond: torch.Tensor, t = 0) -> torch.Tensor:
        """
        Samples new images using the trained diffusion model.
        
        Implements Algorithm 2 from the DDPM paper - the reverse diffusion process.
        Starting from pure noise, gradually denoise to generate new samples.
        
        Args:
            n_sample (int): Number of samples to generate
            size (tuple): Size of each sample
            device: Device to generate samples on
            t (int): Starting timestep (default=0)
            
        Returns:
            torch.Tensor: Generated samples
        """
        # Start from pure noise
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)
    
        # Gradually denoise the samples
        for i in range(self.n_T, t, -1):           
            t_i  = torch.full((n_sample,), i - 1).to(device) / self.n_T  
            epsilon_i = self.eps_model(x_i, t_i, cond) #Use the trained model
            
            x_i = self.oneover_sqrta[i] * (x_i - self.mab_over_sqrtmab[i] * epsilon_i)
            
            if i > 1:
                x_i += torch.randn_like(x_i) * self.sqrt_beta_t[i] #If not at the last step, add noise 

        return x_i