import torch
import torch.nn.functional as F
import torch.nn as nn


def cwgan_discriminator_loss(real_pred, fake_pred, gradients, lambda_gp):
    """
    Compute the CWGAN discriminator loss.

    Args:
        real_pred (Tensor): Discriminator outputs for real samples D(X|Z).
        fake_pred (Tensor): Discriminator outputs for generated samples D(X~|Z).
        gradients (Tensor): Gradients of D w.r.t interpolated samples.
        lambda_gp (float): Gradient penalty coefficient λ1.

    Returns:
        Tensor: Discriminator loss.
    """
    wasserstein_loss = -torch.mean(real_pred) + torch.mean(fake_pred)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    d_loss = wasserstein_loss + lambda_gp * gradient_penalty
    return d_loss

def cwgan_generator_loss(fake_pred, recon_x, real_x, lambda_l1):
    """
    Compute the CWGAN generator loss.

    Args:
        fake_pred (Tensor): Discriminator outputs for generated samples D(X~|Z).
        recon_x (Tensor): Generated/reconstructed samples X~.
        real_x (Tensor): Ground truth samples X.
        lambda_l1 (float): L1 reconstruction loss coefficient λ2.

    Returns:
        Tensor: Generator loss.
    """
    adv_loss = -torch.mean(fake_pred)
    recon_loss = F.l1_loss(recon_x, real_x, reduction="sum")
    g_loss = adv_loss + lambda_l1 * recon_loss
    return g_loss

def beta_vae_loss_function(recon_x, x, mu, logvar, beta=0):

    """
    Compute the beta-VAE loss.
    
    Args:
        recon_x: Reconstructed input.
        x: Original input.
        mu: Mean from the encoder.
        logvar: Log variance from the encoder.
        beta: Weight for KL divergence term.
    
    Returns:
        loss: Computed loss value.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * KLD

def vitae_sl_loss_function(x, dex_x, enc_x, lambda_1=0.8, lambda_2=0.2):
    """Computes the VITAE-SL loss function, weighted sums of MSE for the decoder output Pd and encoder output Pe.
    L= lambda_1 * Ld + lambda_2 * Le
    where Ld is the MSE loss between the decoder output and the input, and Le is the MSE loss between the encoder output and the input.
    The lambda_1 and lambda_2 parameters control the relative importance of the two losses.
    The loss function is used to train the VITAE-SL model.
    The model is trained to minimize the difference between the input and the reconstructed output, while also ensuring that the encoder output is similar to the input.
    The lambda_1 and lambda_2 parameters can be adjusted to control the trade-off between the two losses.

    Args:
        x (_type_): _
        dex_x (_type_): decoder output tensor
        enc_x (_type_): encoder output tensor
        lambda_1 (float, optional): _description_. 
        lambda_2 (float, optional): _description_.
    """
    
    recon_loss = nn.functional.mse_loss(dex_x, x, reduction='mean')
    enc_loss = nn.functional.mse_loss(enc_x, x, reduction='mean')
    return lambda_1 * recon_loss + lambda_2 * enc_loss


def get_loss_function(config):
    """
    Get the loss function based on the configuration.
    
    Args:
        config: Configuration parameters.
    
    Returns:
        criterion: Loss function instance.
    """
    loss = config["loss"]
    
    assert loss in ["mse", "mae", "smoothl1", "vae_elbo", "vitae_sl"], f"Unknown loss function type: {loss}"
    
    if loss == "mse":
        return nn.MSELoss()
    elif loss == "mae":
        return nn.L1Loss()
    elif loss == "smoothl1":
        return nn.SmoothL1Loss()
    elif loss == "vae_elbo":
        return beta_vae_loss_function
    elif loss == "vitae_sl":
        return vitae_sl_loss_function
    elif "cwgan_gp" in loss:
        return None
    elif "diffusion" in loss:
        return None  # Placeholder for diffusion loss function, to be implemented separately
    else:
        raise ValueError(f"Unknown loss function type: {loss}")
