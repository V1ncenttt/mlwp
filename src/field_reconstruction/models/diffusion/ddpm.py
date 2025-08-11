

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
        criterion: nn.Module = nn.MSELoss()
    ) -> None:
        super(DDIM, self).__init__()
        self.eps_model = eps_model

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
        
        This implements Algorithm 1 from the DDPM paper (training is identical to DDPM):
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
    
    def sample(self, n_sample: int, size, device, cond: torch.Tensor, t = 0, eta: float = 0.0, ddim_steps: int = 50) -> torch.Tensor:
        """
        Samples new images using the trained diffusion model with DDIM sampling.
        
        Implements the DDIM sampling algorithm - deterministic sampling for faster generation.
        When eta=0, sampling is completely deterministic. When eta=1, it reduces to DDPM.
        
        Args:
            n_sample (int): Number of samples to generate
            size (tuple): Size of each sample
            device: Device to generate samples on
            cond (torch.Tensor): Conditioning tensor
            t (int): Starting timestep (default=0)
            eta (float): Stochasticity parameter (0=deterministic, 1=stochastic like DDPM)
            ddim_steps (int): Number of sampling steps (can be much smaller than n_T)
            
        Returns:
            torch.Tensor: Generated samples
        """
        # Ensure conditioning is on correct device
        cond = cond.to(device)
        
        # Create a subset of timesteps for faster sampling
        if ddim_steps < self.n_T:
            # Use uniform spacing for timesteps (reverse order for denoising)
            timesteps = torch.linspace(self.n_T-1, t, ddim_steps, dtype=torch.long, device=device)
        else:
            timesteps = torch.arange(self.n_T-1, t, -1, dtype=torch.long, device=device)
        
        # Start from pure noise
        x_i = torch.randn(n_sample, *size, device=device)  # x_T ~ N(0, 1)
    
        # Gradually denoise the samples using DDIM
        for i in range(len(timesteps)):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            
            # Current timestep normalized to [0,1]
            t_i = torch.full((n_sample,), t_curr.float() / self.n_T, device=device)
            
            # Predict noise using the model
            epsilon_i = self.eps_model(x_i, t_i, cond)
            
            # Get alpha_bar values directly from the schedule
            alpha_bar_t = self.alphabar_t[t_curr]
            alpha_bar_t_next = self.alphabar_t[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
            
            # Predict x_0 from current x_t and predicted noise
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            x_0_pred = (x_i - sqrt_one_minus_alpha_bar_t * epsilon_i) / sqrt_alpha_bar_t
            
            # Compute direction pointing towards x_t for next step
            if t_next > 0:
                sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
                sqrt_one_minus_alpha_bar_t_next = torch.sqrt(1 - alpha_bar_t_next)
                
                # DDIM update rule: x_{t-1} = sqrt(ᾱ_{t-1}) * x_0_pred + sqrt(1-ᾱ_{t-1}) * ε_t
                x_i = sqrt_alpha_bar_t_next * x_0_pred + sqrt_one_minus_alpha_bar_t_next * epsilon_i
                
                # Add stochasticity if eta > 0 (makes it more like DDPM)
                if eta > 0:
                    sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)
                    x_i += sigma_t * torch.randn_like(x_i)
            else:
                # Final step: just return the predicted x_0
                x_i = x_0_pred

        return x_i