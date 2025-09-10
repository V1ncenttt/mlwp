

import torch
import torch.distributions as dist
from models.diffusion.noise_schedule import ddpm_schedules
import torch.nn as nn
from typing import Tuple
import math
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
        criterion: nn.Module = nn.MSELoss(),
        cfg_dropout_prob: float = 0.1
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        self.cfg_dropout_prob = cfg_dropout_prob

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)  # Let PyTorch handle device placement automatically

        self.n_T = n_T
        self.criterion = criterion

        # Initialize with dataset statistics for potentially better starting point
        mean =  0 #TODO: make more precise
        std_dev = 1
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
            x (torch.Tensor): Input target data (batch_size, 5, height, width)
            cond (torch.Tensor): Conditioning data (batch_size, 6, height, width)
            t (torch.Tensor, optional): Specific timestep. If None, randomly sampled.
            
        Returns:
            Tuple containing:
            - Loss between predicted and actual noise
            - The sampled noise (eps)
            - The noised input at timestep t (x_t)
        """
         
        # Step 1: Sample timestep t if not provided
        if t is None:
            t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)
            
        # Step 2: Sample noise from normal distribution (only for target data)
        epsilon = self.normal_dist.sample(x.shape).to(x.device) 
        
        # Step 3: Create noised input x_t using the forward process
        # x_t = √(αbar_t) * x_0 + √(1-αbar_t) * ε
        sqrtab_t = self.sqrtab[t].view(-1, 1, 1, 1)
        sqrtmab_t = self.sqrtmab[t].view(-1, 1, 1, 1)
        
        x_t = sqrtab_t * x + sqrtmab_t * epsilon

        # Step 4 & 5: Predict noise using eps_model with separate inputs
        # Pass noised target data and clean conditioning separately
        predicted_epsilon = self.eps_model(x_t, t.float() / self.n_T, cond)
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
        x_i = torch.randn(n_sample, *size, device=device)  # x_T ~ N(0, 1)
    
        # Gradually denoise the samples
        for i in range(self.n_T, t, -1):           
            t_i = torch.full((n_sample,), i - 1, device=device, dtype=torch.float32) / self.n_T  
            epsilon_i = self.eps_model(x_i, t_i, cond) #Use the trained model
            
            x_i = self.oneover_sqrta[i] * (x_i - self.mab_over_sqrtmab[i] * epsilon_i)
            
            if i > 1:
                x_i += torch.randn_like(x_i) * self.sqrt_beta_t[i] #If not at the last step, add noise 

        return x_i

class UncondDDPM(nn.Module):
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
        super(UncondDDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)  # Let PyTorch handle device placement automatically

        self.n_T = n_T
        self.criterion = criterion

        # Initialize with dataset statistics for potentially better starting point
        mean =  0.04640255868434906 #TODO: Replace with dataset mean
        std_dev = 0.8382343649864197
        self.normal_dist = dist.Normal(mean, std_dev)
        
    def set_normal_dist(self, mean: float, std_dev: float) -> None:
        """
        Sets the mean and standard deviation for the noise distribution.
        
        Args:
            mean (float): Mean of the noise distribution
            std_dev (float): Standard deviation of the noise distribution
        """
        self.normal_dist = dist.Normal(mean, std_dev)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
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
            t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)
            
        # Step 2: Sample noise from normal distribution
        epsilon = self.normal_dist.sample(x.shape).to(x.device) 
        
        # Step 3: Create noised input x_t using the forward process
        # x_t = √(αbar_t) * x_0 + √(1-αbar_t) * ε
        sqrtab_t = self.sqrtab[t].view(-1, 1, 1, 1)
        sqrtmab_t = self.sqrtmab[t].view(-1, 1, 1, 1)
        
        x_t = sqrtab_t * x + sqrtmab_t * epsilon

        # Step 4 & 5: Predict noise using eps_model and return loss
        # Note: timesteps are normalized to [0,1] range for the model

        predicted_epsilon = self.eps_model(x_t, t.float() / self.n_T)
        loss = self.criterion(predicted_epsilon, epsilon)
        
        return loss, epsilon, x_t

    def sample(self, n_sample: int, size, device, t = 0) -> torch.Tensor:
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
        x_i = torch.randn(n_sample, *size, device=device)  # x_T ~ N(0, 1)
    
        # Gradually denoise the samples
        for i in range(self.n_T, t, -1):           
            t_i = torch.full((n_sample,), i - 1, device=device, dtype=torch.float32) / self.n_T  
            epsilon_i = self.eps_model(x_i, t_i) #Use the trained model
            
            x_i = self.oneover_sqrta[i] * (x_i - self.mab_over_sqrtmab[i] * epsilon_i)
            
            if i > 1:
                x_i += torch.randn_like(x_i) * self.sqrt_beta_t[i] #If not at the last step, add noise 

        return x_i

class DDIM(nn.Module):
    """
    Denoising Diffusion Implicit Model (DDIM) implementation.
    
    This class implements the DDIM as described in "Denoising Diffusion Implicit Models" 
    (Song et al., 2021). The model learns to reverse a gradual noising process with
    deterministic sampling for faster generation.
    
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
        criterion: nn.Module = nn.MSELoss(),
        cfg_dropout_prob: float = 0.1  # 🆕 CFG dropout probability
    ) -> None:
        super(DDIM, self).__init__()
        self.eps_model = eps_model
        self.cfg_dropout_prob = cfg_dropout_prob  # 🆕 Store CFG parameters

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)  # Let PyTorch handle device placement automatically

        self.eta = 0.0  # Stochasticity parameter for DDIM
        self.ddim_steps = n_T  # Default to full number of steps
        
        self.n_T = n_T
        self.criterion = criterion

        # Initialize with dataset statistics for potentially better starting point
        mean =  0 #TODO: make more precise
        std_dev = 1
        self.normal_dist = dist.Normal(mean, std_dev)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Performs forward diffusion and predicts the noise added at timestep t.
        Now supports Classifier-Free Guidance by randomly dropping conditioning.
        
        Args:
            x (torch.Tensor): Input target data (batch_size, 5, height, width)
            cond (torch.Tensor): Conditioning data (batch_size, 6, height, width)
            t (torch.Tensor, optional): Specific timestep. If None, randomly sampled.
            
        Returns:
            Tuple containing:
            - Loss between predicted and actual noise
            - The sampled noise (eps)
            - The noised input at timestep t (x_t)
        """
         
        # Step 1: Sample timestep t if not provided
        if t is None:
            t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)
            
        # Step 2: Sample noise from normal distribution (only for target data)
        epsilon = self.normal_dist.sample(x.shape).to(x.device) 
        
        # Step 3: Create noised input x_t using the forward process
        # x_t = √(αbar_t) * x_0 + √(1-αbar_t) * ε
        sqrtab_t = self.sqrtab[t].view(-1, 1, 1, 1)
        sqrtmab_t = self.sqrtmab[t].view(-1, 1, 1, 1)
        
        x_t = sqrtab_t * x + sqrtmab_t * epsilon

        # Step 4 & 5: Predict noise using eps_model with CFG dropout during training
        if self.training and self.cfg_dropout_prob > 0:
            # Randomly set conditioning to None for some samples  
            batch_size = cond.shape[0]
            dropout_mask = torch.rand(batch_size, device=cond.device) < self.cfg_dropout_prob
            
            # If we need to handle mixed batches (some with, some without conditioning)
            if dropout_mask.any() and not dropout_mask.all():
                # Split batch into conditional and unconditional
                cond_indices = ~dropout_mask
                uncond_indices = dropout_mask
                
                # Process conditional samples
                if cond_indices.any():
                    x_t_cond = x_t[cond_indices]
                    t_cond = t[cond_indices]
                    cond_cond = cond[cond_indices]
                    pred_eps_cond = self.eps_model(x_t_cond, t_cond.float() / self.n_T, cond_cond)
                
                # Process unconditional samples  
                if uncond_indices.any():
                    x_t_uncond = x_t[uncond_indices]
                    t_uncond = t[uncond_indices]
                    pred_eps_uncond = self.eps_model(x_t_uncond, t_uncond.float() / self.n_T, None)
                
                # Combine predictions back to original order
                predicted_epsilon = torch.zeros_like(epsilon)
                if cond_indices.any():
                    predicted_epsilon[cond_indices] = pred_eps_cond
                if uncond_indices.any():
                    predicted_epsilon[uncond_indices] = pred_eps_uncond
            else:
                # All samples have same conditioning status
                effective_cond = None if dropout_mask.all() else cond
                predicted_epsilon = self.eps_model(x_t, t.float() / self.n_T, effective_cond)
        else:
            # Standard forward pass without CFG dropout
            predicted_epsilon = self.eps_model(x_t, t.float() / self.n_T, cond)
        
        loss = self.criterion(predicted_epsilon, epsilon)
        
        return loss, epsilon, x_t
    
    def set_eta(self, eta: float) -> None:
        """
        Sets the stochasticity parameter for DDIM sampling.
        
        Args:
            eta (float): Stochasticity parameter (0=deterministic, 1=stochastic like DDPM)
        """
        self.eta = eta
        
    def set_ddim_steps(self, ddim_steps: int) -> None:
        """
        Sets the number of DDIM sampling steps.
        
        Args:
            ddim_steps (int): Number of sampling steps (can be much smaller than n_T)
        """
        self.ddim_steps = ddim_steps
    
    def get_optimized_timesteps(self, n_T, ddim_steps, schedule='linear', min_t=10):
        """Safe timestep scheduling with minimum timestep protection"""
        
        if schedule == 'cosine':
            # SAFE cosine schedule - avoid very small timesteps
            steps = torch.linspace(0, 1, ddim_steps + 1)[1:]  # Skip t=0, range (0, 1]
            steps = (torch.cos(steps * math.pi) + 1) / 2  # Maps to [~0.85, 0.0] → we want to avoid 0.0
            steps = steps * (n_T - min_t) + min_t  # Scale to [min_t, n_T-1]
            
        elif schedule == 'sqrt':
            # SAFE square root schedule
            steps = torch.linspace(0, 1, ddim_steps + 1)[1:]  # Skip t=0
            steps = torch.sqrt(steps) * (n_T - min_t) + min_t  # [min_t, n_T-1]
            
        else:  # linear (default) - always safe
            steps = torch.linspace(min_t, n_T-1, ddim_steps)  # Start from min_t
        
        steps = torch.round(steps).long()
        steps = torch.unique_consecutive(steps)
        
        # Final safety check - ensure no timesteps below min_t
        steps = steps[steps >= min_t]
        
        return steps.flip(0)  # Descending order


    def sample(self, n_sample: int, size, device, cond: torch.Tensor, 
           t: int = 0, eta: float = 0.0, ddim_steps: int = 100, 
           cfg_scale: float = 1, noise_aware_cfg: bool = False) -> torch.Tensor:
        cond = cond.to(device)

        # robust, unique, descending timestep schedule
        ts = torch.linspace(0, self.n_T - 1, steps=ddim_steps, device=device)
        # Usage in sample():
        #timesteps = self.get_optimized_timesteps(self.n_T, ddim_steps, schedule='sqrt')
        timesteps = torch.round(ts).long().flip(0)
        timesteps = torch.unique_consecutive(timesteps)
        print("Sampling with " + str(ddim_steps) + " steps") 
        x = torch.randn(n_sample, *size, device=device)

        for idx, t_curr in enumerate(timesteps):
            t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else torch.tensor(-1, device=device)
            a_t = self.alphabar_t[t_curr]                       # scalar tensor
            a_prev = torch.tensor(1.0, device=device) if t_prev.item() < 0 else self.alphabar_t[t_prev]

            sqrt_a_t = torch.sqrt(a_t)
            sqrt_1ma_t = torch.sqrt(1 - a_t)

            # normalized t input
            t_in = torch.full((n_sample,), float(t_curr.item()) / self.n_T, device=device)

            # classifier-free guidance (use small scales for fields)
            eps_u = self.eps_model(x, t_in, None)
            if cfg_scale > 1e-6:
                eps_c = self.eps_model(x, t_in, cond)
                if noise_aware_cfg:
                    # see §2 for a better schedule
                    w_t = cfg_scale * (1.0 + (1 - a_t))  # Signal-to-noise ratio
                else:
                    w_t = torch.tensor(cfg_scale, device=device)
                eps_hat = eps_u + w_t * (eps_c - eps_u)
            else:
                eps_hat = eps_u

            # predict x0
            x0_hat = (x - sqrt_1ma_t * eps_hat) / sqrt_a_t

            # DDIM update with correct variance split
            # sigma_t = 0 for deterministic (eta=0)
            sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
            # direction term keeps the remaining variance (minus sigma_t^2)
            dir_coeff = torch.sqrt(torch.clamp(1 - a_prev - sigma_t**2, min=0.0))

            x = torch.sqrt(a_prev) * x0_hat + dir_coeff * eps_hat
            if eta > 0:
                x = x + sigma_t * torch.randn_like(x)

        return x