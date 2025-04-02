import torch
import torch.optim as optim
import torch.nn as nn
from models import FukamiNet
from utils import get_device

def create_model(model):
    if model == "fukami":
        return FukamiNet()
    elif model == "VAE":
        # Assuming VAE is defined elsewhere
        return None
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
    if config.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer}")   

def get_loss_function(config):
    """
    Get the loss function based on the configuration.
    
    Args:
        config: Configuration parameters.
    
    Returns:
        criterion: Loss function instance.
    """
    if config.loss == "mse":
        return nn.MSELoss()
    elif config.loss == "mae":
        return nn.L1Loss()
    elif config.loss == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function type: {config.loss}")
    
def train(model, data, model_save_path, config):
    """
    Train the model on the given data.
    
    Args:
        model: Model to train.
        data: Training data.
        model_save_path: Path to save the trained model.
        config: Configuration parameters.
    """
    device = get_device()
    model = create_model(model).to(device)
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    folds = config.folds
    train_loader = data.train_loader
    val_loader = data.val_loader
    
    
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)