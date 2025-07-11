import torch

class KarrasSigmasLognormal:
    
    def __init__(self, sigmas, P_mean=1.2, P_std=1.2, sigma_min=0.002, sigma_max=80.0, nb_timesteps=1000):
        self.sigmas = sigmas
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.nb_timesteps = nb_timesteps
        
    def __call__(self, batch_size, generator=None, device="cuda"):
        """
        Generate a batch of noise samples based on the Karras lognormal distribution.
        
        Args:
            batch_size (int): The number of samples to generate.
            generator (torch.Generator, optional): A PyTorch generator for reproducibility.
            device (str): The device to generate the samples on.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, nb_timesteps) containing the noise samples.
        """
        random_normal = torch.randn([batch_size, 1, 1, 1], device=device, generator=generator)
        sigma = (random_normal * self.P_std + self.P_mean).exp()
        
        sigmas_expanded = self.sigmas[:-1].view(1, -1).to(device)
        sigma_expanded = sigma.view(batch_size, 1)
        
        difference = torch.abs(sigma_expanded - sigmas_expanded)
        indices = torch.argmin(difference, dim=1)
        
        return indices
