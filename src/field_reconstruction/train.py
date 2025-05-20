import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import random
import numpy as np
from utils import get_device, create_model
import matplotlib.pyplot as plt
from plots_creator import plot_voronoi_reconstruction_comparison, plot_random_reconstruction


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
    
    assert optimizer in ["adam", "adamw", "sgd"], f"Unknown optimizer type: {optimizer}"
    assert lr > 0, f"Learning rate must be positive: {lr}"
    assert weight_decay >= 0, f"Weight decay must be non-negative: {weight_decay}"
   
    if optimizer == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer}")   

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
    else:
        raise ValueError(f"Unknown loss function type: {loss}")



                 
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
    
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    
    sample_input, _ = next(iter(train_loader))
    nb_channels = sample_input.shape[1]
    
    model = create_model(model_name, nb_channels=nb_channels).to(device)
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    epochs = config["epochs"]

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
            print(f"Inputs shape: {inputs.shape}")
            print(f"Targets shape: {targets.shape}")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if model_name == "vae" and config["loss"] == "vae_elbo":
                recon_x, mu, logvar = model(inputs)
                loss = criterion(recon_x, targets, mu, logvar)
            elif "vitae" in model_name and config["loss"] == "vitae_sl":
                dex_x, enc_x = model(inputs)
                loss = criterion(targets, dex_x, enc_x)
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
                elif "vitae" in model_name and config["loss"] == "vitae_sl":
                    dex_x, enc_x = model(inputs)
                    print(f"Decoder output shape: {dex_x.shape}")
                    print(f"Encoder output shape: {enc_x.shape}")
                    loss = criterion(targets, dex_x, enc_x)
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
