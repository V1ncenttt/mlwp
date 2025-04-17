import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import random
import numpy as np
from models import FukamiNet, ReconstructionVAE
from utils import get_device
from plots_creator import plot_voronoi_reconstruction_comparison

def create_model(model):
    if model == "fukami":
        return FukamiNet()
    elif model == "vae":
        # Assuming VAE is defined elsewhere
        return ReconstructionVAE(channels=2, latent_dim=128)
    else:
        raise ValueError(f"Unknown model type: {model}")

def get_optimizer(model, config):
    """
    Get the optimizer for the model.
    
    Args:
        model: Model to optimize.
        config: Configuration parameters.
    
    Returns:
        optimizer: Optimizer instance.
    """
    optimizer = config["optimizer"]
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer}")   

def beta_vae_loss_function(recon_x, x, mu, logvar, beta=2):

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

def get_loss_function(config):
    """
    Get the loss function based on the configuration.
    
    Args:
        config: Configuration parameters.
    
    Returns:
        criterion: Loss function instance.
    """
    loss = config["loss"]
    if loss == "mse":
        return nn.MSELoss()
    elif loss == "mae":
        return nn.L1Loss()
    elif loss == "smoothl1":
        return nn.SmoothL1Loss()
    elif loss == "vae_elbo":
        return beta_vae_loss_function
    else:
        raise ValueError(f"Unknown loss function type: {loss}")

def plot_random_reconstruction(model, val_loader, device, model_name, save_dir):
    """
    Run the trained model on a random validation sample and plot the output.
    """
    model.eval()
    with torch.no_grad():
        sample_idx = random.randint(0, len(val_loader.dataset) - 1)
        x, y = val_loader.dataset[sample_idx]  # single sample
        x = x.unsqueeze(0).to(device)          # add batch dim
        y = y.squeeze().cpu().numpy()          # (1, H, W) → (H, W)

        if model_name == "vae":
            recon_x, mu, logvar = model(x)
            pred = recon_x.squeeze().cpu().numpy()
        else:
            pred = model(x).squeeze().cpu().numpy()   # (1, H, W) → (H, W)

        x_np = x.squeeze().cpu().numpy()          # (2, H, W)
        tess = x_np[0]  # Voronoi tessellated field
        mask = x_np[1]  # sensor mask (0/1)

        plot_path = os.path.join(save_dir, f"{model_name}_reco_sample.png")
        plot_voronoi_reconstruction_comparison(
            voronoi_mask=mask,
            ground_truth=y,
            cnn_output=pred,
            mask=mask,
            save_path=plot_path
        )
           
def train(model_name, data, model_save_path, config):
    """
    Train the model on the given data.

    Args:
        model_name: Name of the model (str).
        data: Dict with 'train_loader' and 'val_loader'.
        model_save_path: Path to save the trained model.
        config: Configuration parameters (dict).
    """
    device = get_device()
    model = create_model(model_name).to(device)
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    epochs = config["epochs"]
    
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]

    print(f"📊 Number of training samples: {len(train_loader.dataset)}")
    print(f"📊 Number of validation samples: {len(val_loader.dataset)}")
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"📁 Created directory: {model_save_path}")
        
    print("📦 Training started...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if model_name == "vae" and config["loss"] == "vae_elbo":
                recon_x, mu, logvar = model(inputs)
                loss = criterion(recon_x, targets, mu, logvar)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]", leave=False)
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                if model_name == "vae" and config["loss"] == "vae_elbo":
                    recon_x, mu, logvar = model(inputs)
                    loss = criterion(recon_x, targets, mu, logvar)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"📉 Epoch {epoch}/{epochs} — Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save model
    if model_save_path:
        complete_path = os.path.join(model_save_path, f"{model_name}_model_{epochs}_{val_loss:.6f}.pth")
        torch.save(model.state_dict(), complete_path)
        print(f"💾 Model saved to {complete_path}")
    
    # Save last validation loss to file
    with open(os.path.join(model_save_path, f"last_loss_{epochs}.txt"), "w") as f:
        f.write(f"Last validation loss: {val_loss:.6f}")
    
    plot_random_reconstruction(model, val_loader, device, model_name, model_save_path)

    return model
