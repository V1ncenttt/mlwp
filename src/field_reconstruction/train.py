import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import random
import numpy as np
from utils import get_device, create_model
import matplotlib.pyplot as plt
from plots_creator import plot_voronoi_reconstruction_comparison


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


def plot_random_reconstruction(model, val_loader, device, model_name, save_dir, num_samples=7):
    """
    Plot reconstruction vs ground truth for multiple variables/channels.
    For each variable, generate a 7x3 grid: Ground Truth, Prediction, Error.
    """
    model.eval()
    dataset = val_loader.dataset
    total_samples = len(dataset)
    input_sample, target_sample = dataset[0]
    num_channels = target_sample.shape[0]

    with torch.no_grad():
        for ch in range(num_channels):
            fig, axs = plt.subplots(num_samples, 3, figsize=(12, 2.2 * num_samples))
            fig.suptitle(f"Channel {ch}: Variable {ch}", fontsize=14)

            for i in range(num_samples):
                idx = random.randint(0, total_samples - 1)
                x, y = dataset[idx]
                x = x.unsqueeze(0).to(device)
                y = y[ch]  # shape (H, W)

                if model_name == "vae":
                    recon_x, mu, logvar = model(x)
                    pred = recon_x.squeeze().cpu().numpy()[ch]
                else:
                    pred = model(x).squeeze().cpu().numpy()[ch]

                gt = y.cpu().numpy()
                error = np.abs(gt - pred)

                axs[i, 0].imshow(gt, cmap='viridis')
                axs[i, 0].set_title(f"Image {i} — Ground Truth")
                axs[i, 0].axis("off")

                axs[i, 1].imshow(pred, cmap='viridis')
                axs[i, 1].set_title("Reconstruction")
                axs[i, 1].axis("off")

                axs[i, 2].imshow(error, cmap='hot')
                axs[i, 2].set_title("Abs Error")
                axs[i, 2].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            fname = os.path.join(save_dir, f"{model_name}_reco_channel_{ch}.png")
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"📸 Saved plot for channel {ch} to {fname}")   
                 
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
