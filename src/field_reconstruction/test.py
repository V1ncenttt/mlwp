

import torch
import os
import numpy as np
from tqdm import tqdm
from models import FukamiNet, ReconstructionVAE
from utils import get_device
from plots_creator import plot_voronoi_reconstruction_comparison

def create_model(model_name, channels=2, latent_dim=128):
    """_summary_

    Args:
        model_name (_type_): _description_
        channels (int, optional): _description_. Defaults to 2.
        latent_dim (int, optional): _description_. Defaults to 128.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if model_name == "fukami":
        return FukamiNet()
    elif model_name == "vae":
        return ReconstructionVAE(channels=channels, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def rrmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)) / torch.sqrt(torch.mean(target ** 2))

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def evaluate(model, test_loader, model_name, config):
    device = get_device()
    model = create_model(model_name).to(device)
    
    # Load checkpoint
    checkpoint_path = config["checkpoint_path"]
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    rrmse_total = 0.0
    mae_total = 0.0
    n = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if model_name == "vae":
                recon_x, mu, logvar = model(inputs)
                preds = recon_x
            else:
                preds = model(inputs)

            rrmse_batch = rrmse(preds, targets).item()
            mae_batch = mae(preds, targets).item()
            
            rrmse_total += rrmse_batch * inputs.size(0)
            mae_total += mae_batch * inputs.size(0)
            n += inputs.size(0)

            pbar.set_postfix({"RRMSE": rrmse_batch, "MAE": mae_batch})

    rrmse_avg = rrmse_total / n
    mae_avg = mae_total / n
    print(f"✅ Test RRMSE: {rrmse_avg:.6f}")
    print(f"✅ Test MAE:   {mae_avg:.6f}")

    return {"rrmse": rrmse_avg, "mae": mae_avg}