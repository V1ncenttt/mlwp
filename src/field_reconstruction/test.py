

import torch
import os
import numpy as np
from tqdm import tqdm
from models import FukamiNet, ReconstructionVAE
from utils import get_device
from plots_creator import plot_voronoi_reconstruction_comparison
from utils import create_model, get_device

def rrmse(pred, target):
    """
    The function calculates the relative root mean square error between two input tensors.
    
    :param pred: The `pred` parameter typically refers to the predicted values from a model, while the
    `target` parameter refers to the actual target values. The function `rrmse` seems to be calculating
    the Root Relative Mean Squared Error (RRMSE) between the predicted values and the target values
    :param target: The `target` parameter typically refers to the ground truth values or the actual
    values that you are trying to predict or model. In the context of the `rrmse` function you provided,
    `target` is likely the true target values that you are comparing against the predicted values
    (`pred`)
    :return: the Relative Root Mean Squared Error (RRMSE) between the predicted values (pred) and the
    target values.
    """
    
    return torch.sqrt(torch.mean((pred - target) ** 2)) / torch.sqrt(torch.mean(target ** 2))

def mae(pred, target):
    """
    The function calculates the mean absolute error between two input tensors.
    
    :param pred: The `pred` parameter typically refers to the predicted values generated by a model,
    while the `target` parameter refers to the actual target values that the model is trying to predict.
    The `mae` function calculates the Mean Absolute Error (MAE) between the predicted values (`pred`)
    and the
    :param target: The `target` parameter typically refers to the true or actual values that you are
    trying to predict or estimate in a machine learning model. It is the ground truth against which your
    predictions are compared
    :return: The function `mae` returns the mean absolute error between the `pred` and `target` tensors
    using the torch library.
    """
    return torch.mean(torch.abs(pred - target))

def evaluate(model_type, test_loader,checkpoint_path):
    device = get_device()
    model = create_model(model_type).to(device)
    
    # Load checkpoint
    #Add folder to the path
    checkpoint_path = os.path.join("models/saves", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"✅ Evaluating model on {len(test_loader.dataset)} samples")
    print(f"✅ Model type: {model_type}")
    print(f"✅ Device: {device}")
    rrmse_total = 0.0
    mae_total = 0.0
    n = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if model_type == "vae":
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
    print(f"Evaluation results for {checkpoint_path}:")
    print(f"✅ Test RRMSE: {rrmse_avg:.6f}")
    print(f"✅ Test MAE:   {mae_avg:.6f}")

    return {"rrmse": rrmse_avg, "mae": mae_avg}