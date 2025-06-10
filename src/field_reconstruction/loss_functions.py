import torch
import torch.nn.functional as F

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
    recon_loss = F.l1_loss(recon_x, real_x)
    g_loss = adv_loss + lambda_l1 * recon_loss
    return g_loss
