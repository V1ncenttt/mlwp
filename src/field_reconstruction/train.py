import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import random
import numpy as np
from utils import get_device, create_model
import matplotlib.pyplot as plt
#import mlflow
#import mlflow.pytorch
#from mlflow.entities import Dataset
import torchvision
import wandb
import os
try:
    from coolname import generate_slug
except ImportError:
    print("⚠️ coolname not installed. Install with: pip install coolname")
    def generate_slug(words=2):
        adjectives = ["swift", "bold", "clever", "bright", "calm", "wild", "quiet", "brave"]
        animals = ["panda", "tiger", "wolf", "eagle", "fox", "bear", "deer", "lion"]
        return f"{random.choice(adjectives)}-{random.choice(animals)}"
    
from plots_creator import plot_voronoi_reconstruction_comparison, plot_random_reconstruction
from loss_functions import cwgan_discriminator_loss, cwgan_generator_loss, get_loss_function
from models.diffusion.diffusion_unet import SimpleUnet, UnconditionalUnet
from models.diffusion.ddpm import DDPM, UncondDDPM, DDIM
from models.diffusion.ema_weights import EMA
from models.fukami import FukamiNet

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_experiment("diffusion-fieldreco")
wandb.login()
run = None

def initialize_wandb(model_name, config):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        model_name: Name of the model (str).
        config: Configuration parameters (dict).
    """
    global run
    
    print("Initializing W&B run...")
    run = wandb.init(
        entity="vincent-lfve-imperial-college-london",
        project="MLWP",
        name=model_name,
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "optimizer": config["optimizer"],
            "loss": config["loss"]
        },
        reinit=True
    )
    print(f"Initialized W&B run: {run.id} for model: {model_name}")
    return run

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
    
    
    if "cwgan" in model_name.lower():
        model = create_model(model_name, nb_channels=nb_channels-1)
        print("⚠️  Detected CWGAN model — using train_cwgan() instead of train()")
        generator, discriminator = model
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        train_cwgan(generator, discriminator, data, model_save_path, config)
        return  # Exit normal train() after calling train_cwgan
    elif "diffusion" in model_name.lower():
        print("⚠️  Detected Diffusion model — using train_diffusion_model() instead of train()")
        # Create model with 5 input channels (target data only) 
        # Conditioning will be handled separately in the new architecture
        model = create_model(model_name, nb_channels=5)  # Only target channels
        print(f"🟢 Diffusion model created: {model_name} with 5 input channels and separate conditioning")
        train_diffusion_model(model, data, model_save_path, config)
        return
    else:
        model = create_model(model_name, nb_channels=nb_channels)
        print(f"🟢 Model created: {model_name} with input channels: {nb_channels}")
    
    model = model.to(device)
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
            #print(f"Inputs shape: {inputs.shape}")
            #print(f"Targets shape: {targets.shape}")
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
                    #print(f"Decoder output shape: {dex_x.shape}")
                    #print(f"Encoder output shape: {enc_x.shape}")
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
    
    #plot_random_reconstruction(model, val_loader, device, model_name, model_save_path)

    return model

def train_cwgan(generator, discriminator, data, model_save_path, config):
    device = get_device()
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    nb_channels = next(iter(train_loader))[0].shape[1] -1  # Get number of input channels from first batch
    #Show input and output shapes of generator
    print(f"🟢 Generator input channels: {nb_channels+1}")
    print(f"🟢 Generator output channels: {generator.final.out_channels if hasattr(generator, 'final') else 'Unknown'}")
    #Print lr
    print(f"learning_rate: {config['learning_rate']}")
    lambda_gp = config.get("lambda_gp", 10)
    lambda_l1 = config.get("lambda_l1", 100)
    epochs = config["epochs"]
    critic_iter = config.get("critic_iter", 5)

    sample_input, _ = next(iter(train_loader))
    sample_targets = next(iter(train_loader))[1]
    check_normalization(sample_input, "Sample Inputs")
    check_normalization(sample_targets, "Sample Targets")
    
    generator.to(device)
    discriminator.to(device)

    optimizer_G = get_optimizer(generator, config)
    optimizer_D = get_optimizer(discriminator, config)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"📁 Created directory: {model_save_path}")

    print(f"📊 Number of training samples: {len(train_loader.dataset)}")
    print(f"📊 Number of validation samples: {len(val_loader.dataset)}")
    print("📦 Training started...")

    d_real_loss_history = []
    d_fake_loss_history = []
    grad_penalty_history = []
    g_l1_loss_history = []
    g_adv_loss_history = []

    g_loss_epoch_history = []
    d_loss_epoch_history = []

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_loss_total, d_loss_total = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]", leave=False)
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # In VITAE mode (our case), inputs[0] is mask, inputs[1:] are the 5 fields with 0 elsewhere
            fields_only = inputs[:, 1:, :, :]  # (batch, 5, H, W)

            # === Train Discriminator ===
            for _ in range(critic_iter):
                fields_only_random = torch.cat([fields_only, torch.randn_like(fields_only[:, :1])], dim=1)
                # Now input to Generator is shape: (B, 6, 64, 32)
                z_fake = generator(fields_only_random).detach()
                real_input = torch.cat([fields_only, targets], dim=1)
                fake_input = torch.cat([fields_only, z_fake], dim=1)

                real_pred = discriminator(real_input)
                fake_pred = discriminator(fake_input)
                
                alpha = torch.rand(targets.size(0), 1, 1, 1).to(device)
                interpolates = (alpha * targets + (1 - alpha) * z_fake).requires_grad_(True)
                interpolate_input = torch.cat([fields_only, interpolates], dim=1).requires_grad_(True)
                d_interpolates = discriminator(interpolate_input)

                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolate_input,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                d_loss = cwgan_discriminator_loss(real_pred, fake_pred, gradients, lambda_gp)
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Weight clipping for discriminator (WGAN original trick)
                """
                clip_value = 0.01  # you can adjust this hyperparameter c
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)
                """
                
                d_real_loss = -real_pred.mean().item()
                d_fake_loss = fake_pred.mean().item()
                grad_penalty = gradients.view(gradients.size(0), -1).norm(2, dim=1).sub(1).pow(2).mean().item()
                grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1).mean().item()
                #print(f"Gradient Norm: {grad_norm:.6f}")

                d_real_loss_history.append(d_real_loss)
                d_fake_loss_history.append(d_fake_loss)
                grad_penalty_history.append(grad_penalty)

            # === Train Generator ===
            fields_only_random = torch.cat([fields_only, torch.randn_like(fields_only[:, :1])], dim=1)
            z_fake = generator(fields_only_random)
            fake_input = torch.cat([fields_only, z_fake], dim=1)
            fake_pred = discriminator(fake_input)
            g_loss = cwgan_generator_loss(fake_pred, z_fake, targets, lambda_l1)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            g_l1_loss = nn.functional.l1_loss(z_fake, targets).item()
            g_adv_loss = -fake_pred.mean().item()

            g_l1_loss_history.append(g_l1_loss)
            g_adv_loss_history.append(g_adv_loss)

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            pbar.set_postfix({"G Loss": g_loss.item(), "D Loss": d_loss.item()})

        g_loss_total /= len(train_loader)
        d_loss_total /= len(train_loader)
        g_loss_epoch_history.append(g_loss_total)
        d_loss_epoch_history.append(d_loss_total)
        print(f"📉 Epoch {epoch}/{epochs} — G Loss: {g_loss_total:.6f} | D Loss: {d_loss_total:.6f}")

    torch.save(generator.state_dict(), os.path.join(model_save_path, f"generator_model_{epochs}_{g_loss_total:.6f}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_save_path, f"discriminator_model_{epochs}_{d_loss_total:.6f}.pth"))
    print(f"💾 Generator and Discriminator saved to {model_save_path}")

    # Plot overall G Loss and D Loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(g_loss_epoch_history, label='G Loss')
    plt.plot(d_loss_epoch_history, label='D Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'loss_over_epochs.png'))
    print(f"📊 Loss over epochs plot saved to {model_save_path}/loss_over_epochs.png")

    show_random_cwgan_reconstruction(generator, val_loader, device, model_save_path)

def show_random_cwgan_reconstruction(generator, val_loader, device, save_path):
    """
    Display and save one random reconstruction from the validation loader using the trained generator.

    Args:
        generator: Trained generator model.
        val_loader: Validation data loader.
        device: Computation device.
        save_path: Path to save the reconstruction image.
    """
    generator.eval()
    with torch.no_grad():
        # Get one random batch from val_loader
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)

        # In VITAE mode, inputs[0] is mask, inputs[1:] are the 5 fields with 0 elsewhere
        fields_only = inputs[:, 1:, :, :]

        # Generate reconstruction
        fields_only_random = torch.cat([fields_only, torch.randn_like(fields_only[:, :1])], dim=1)
        reconstruction = generator(fields_only_random)

        # Move tensors to cpu and convert to numpy
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        reconstruction_np = reconstruction.cpu().numpy()

        # Plot original target and reconstruction side by side for a random sample in batch
        idx = random.randint(0, inputs_np.shape[0] - 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        vmax = max(targets_np[idx].max(), reconstruction_np[idx].max())
        vmin = min(targets_np[idx].min(), reconstruction_np[idx].min())

        channel_to_plot = 0  # Plot only the first channel

        axes[0].imshow(targets_np[idx, channel_to_plot], vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Original Target - Channel {channel_to_plot}')
        axes[0].axis('off')

        axes[1].imshow(reconstruction_np[idx, channel_to_plot], vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Generator Reconstruction - Channel {channel_to_plot}')
        axes[1].axis('off')

        plt.suptitle('CWGAN Random Reconstruction')
        plt.tight_layout()
        save_file = os.path.join(save_path, 'random_cwgan_reconstruction.png')
        plt.savefig(save_file)
        plt.close()
        print(f"📊 Random reconstruction saved to {save_file}")

def check_normalization(tensor, name="Tensor"):
    """
    Print min, max, mean, std to check normalization.
    """
    tensor_np = tensor.cpu().numpy()
    print(f"\n{name} stats:")
    print(f"  Min:  {tensor_np.min():.4f}")
    print(f"  Max:  {tensor_np.max():.4f}")
    print(f"  Mean: {tensor_np.mean():.4f}")
    print(f"  Std:  {tensor_np.std():.4f}")

def compute_rrmse(predictions, targets):
    """
    Compute Relative Root Mean Square Error (RRMSE) using standard definition.
    RRMSE = RMSE / RMS(targets) = sqrt(MSE) / sqrt(mean(targets^2))
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth targets
        
    Returns:
        float: RRMSE value
    """
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    target_rms = torch.sqrt(torch.mean(targets ** 2))  # Root Mean Square of targets
    rrmse = rmse / target_rms if target_rms > 0 else rmse
    
    # Debug prints
    print(f"  Debug - MSE: {mse.item():.6f}, RMSE: {rmse.item():.6f}, Target RMS: {target_rms.item():.6f}")
    print(f"  Debug - Pred range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"  Debug - Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
    
    return rrmse.item()

def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error (MAE).
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth targets
        
    Returns:
        float: MAE value
    """
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()
# --- add this helper near your other utils ---
def compute_val_noise_mse(ddim_model, val_loader, device, max_batches=None):
    """
    Compute the DDIM/DDPM training loss (noise MSE) on the validation set.

    Args:
        ddim_model: your DDIM instance (with forward returning (loss, ...))
        val_loader: DataLoader yielding (inputs=cond, targets=x)
        device: torch device
        max_batches: optional int to cap validation cost (None = full val set)

    Returns:
        mean_loss: float, mean MSE over all val samples (weighted by batch size)
    """
    ddim_model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for b_idx, (inputs, targets) in enumerate(val_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            cond = inputs.to(device)
            x = targets.to(device)
            t = torch.randint(0, ddim_model.n_T, (x.shape[0],), device=device)
            loss, _, _ = ddim_model(x, cond, t)  # same objective as training
            # Assume 'loss' is mean over batch. Weight by batch size.
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs
    return total_loss / max(total_count, 1)

def validate_diffusion_model(ddpm, val_loader, device, num_samples=100, num_predictions=5):
    """
    Validate the diffusion model by generating multiple samples and computing metrics.
    
    Args:
        ddpm: Trained diffusion model
        val_loader: Validation data loader
        device: Device to run computations on
        num_samples: Number of validation samples to evaluate
        num_predictions: Number of predictions to generate per sample and average
        
    Returns:
        dict: Dictionary containing validation metrics
    """
    ddpm.eval()
    
    total_rrmse = 0.0
    total_mae = 0.0
    num_batches = 0
    samples_processed = 0
    
    print(f"📊 Generating {num_predictions} predictions per sample for robust validation...")
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            if samples_processed >= num_samples:
                break
                
            # Move to device - separate conditioning and targets
            cond = inputs.to(device)     # Conditioning data (batch, 6, H, W)
            targets = targets.to(device) # Target data (batch, 5, H, W)
            
            # Generate predictions using DDIM sampling
            batch_size = min(targets.shape[0], num_samples - samples_processed)
            if batch_size < targets.shape[0]:
                cond = cond[:batch_size]
                targets = targets[:batch_size]
            
            # Generate multiple predictions and average them
            predictions_list = []
            for pred_idx in range(num_predictions):
                # Sample from the diffusion model
                if isinstance(ddpm, DDIM):
                    single_prediction = ddpm.sample(
                        n_sample=batch_size,
                        size=(targets.shape[1], targets.shape[2], targets.shape[3]),
                        device=device,
                        cond=cond,
                        eta=0.0,  # Deterministic sampling for validation
                        ddim_steps=100,  # Fewer steps for faster validation
                        cfg_scale = 1.5
                    )
                else:  # Default to DDPM sampling
                    single_prediction = ddpm.sample(
                        n_sample=batch_size,
                        size=(targets.shape[1], targets.shape[2], targets.shape[3]),
                        device=device,
                        cond=cond,
                    )
                predictions_list.append(single_prediction)
                
            # Average the predictions
            predictions = torch.stack(predictions_list, dim=0).mean(dim=0)
            
            # Compute metrics for this batch
            batch_rrmse = compute_rrmse(predictions, targets)
            batch_mae = compute_mae(predictions, targets)
            
            total_rrmse += batch_rrmse
            total_mae += batch_mae
            num_batches += 1
            samples_processed += batch_size
            
            print(f"  Validated {samples_processed}/{num_samples} samples...")
    
    # Average metrics across all batches
    avg_rrmse = total_rrmse / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
    
    print(f"✅ Validation complete with {num_predictions} predictions per sample")
    
    return {
        'val_rrmse': avg_rrmse,
        'val_mae': avg_mae,
        'samples_evaluated': samples_processed,
        'predictions_per_sample': num_predictions
    }
    
def train_diffusion_model(model, data, model_save_path, config):
    """
    Train a diffusion model using the specified schedule and method.
    
    Args:
        model: Diffusion model instance.
        data: Dict with 'train_loader' and 'val_loader'.
        schedule: Noise schedule configuration.
        method: Training method (e.g., 'ddpm', 'ddim').
        model_save_path: Path to save the trained model.
        config: Configuration parameters (dict).
    """
    global run
    
    # Generate a unique model name
    model_slug = generate_slug(3)
    model_name = f"ddpm-{model_slug}"
    
    run = initialize_wandb(model_name=model_name, config=config)
    
    device = get_device()
    rn_rn_model = model.to(device)
    batch_size = config.get("batch_size", 64)
    epochs = config["epochs"]
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]

    # Set T from config or default to 50
    T = config.get("timesteps", 1000)

    print("Initializing DDIM model and optimizer...")
    lr = 1e-4
    ddpm = DDIM(rn_rn_model, (1e-4, 0.02), T, cfg_dropout_prob=0.1).to(device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    ema_decay = config.get("ema_decay", 0.9999)
    ema_ramp  = config.get("ema_ramp", 2000)
    #ema = EMA(ddpm, decay=ema_decay, ramp=ema_ramp).to(device)
    #run.log({"ema_decay": ema_decay, "ema_ramp": ema_ramp})

    # Add cosine annealing with warm restarts scheduler
    T_0 = max(1, epochs // 4)  # Initial restart period (every 25% of total epochs)
    T_mult = 2  # Multiply restart period by this factor after each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, 
        T_0=T_0, 
        T_mult=T_mult, 
        eta_min=lr,
    )
    print(f"📈 Added cosine annealing warm restarts scheduler: lr={lr:.2e} -> eta_min={lr:.2e}")
    print(f"📈 Restart schedule: T_0={T_0}, T_mult={T_mult} (restarts every {T_0}, {T_0*T_mult}, {T_0*T_mult**2}, ... epochs)")
    conditioning_mode = rn_rn_model.conditioning_type
    # Log key hyperparameters to W&B
    run.log({
        "model_name": model_name,
        "model_slug": model_slug,
        "initial_learning_rate": lr,
        "sampling":"ddpm",  # Indicate DDPM sampling method
        "timesteps_T": T,
        "batch_size": batch_size,
        "conditioning": conditioning_mode,  # Separate conditioning input
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingWarmRestarts",
        "scheduler_T_0": T_0,
        "scheduler_T_mult": T_mult,
        "min_learning_rate": lr,
        "total_epochs": epochs,
        "cfg_dropout_prob": ddpm.cfg_dropout_prob,
        "cfg_scale": 2.0
    })
    print(f"📊 Logged hyperparameters - Model: {model_name}, LR: {lr:.2e}, T: {T}, Batch Size: {batch_size}")

    # Get latent_shape from first batch (target data only)
    sample_batch = next(iter(train_loader))
    latent_shape = sample_batch[1].shape  # targets shape (batch, 5, H, W)


    lastepoch = 0

    ddpm.to(device)


    for i in range(lastepoch, epochs):
        ddpm.train()
        rn_rn_model.train()
        batch_pbar = tqdm(train_loader, leave=False)
        loss_ema = None

        # Training loop
        for inputs, targets in batch_pbar:
            # Separate conditioning and target data for new UNet architecture
            cond = inputs.to(device)  # Full conditioning input (batch, 6, H, W)
            x = targets.to(device)    # Target data only (batch, 5, H, W)
            
            t = torch.randint(0, ddpm.n_T, (x.shape[0],), device=device) #Sample a random timestep
            loss, _, _ = ddpm(x, cond, t)  # Pass x and cond separately
            loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()
            
            # Log individual batch loss (like before)
            #print(f"Epoch {i} | Loss: {loss.item():.4f} | Loss EMA: {loss_ema:.4f}")
            batch_pbar.set_description(f"Epoch {i} | Loss EMA: {loss_ema:.4f}")
            
            # Log to W&B every batch (like before)
            run.log({"train_loss": loss.item(), "train_loss_ema": loss_ema, "epoch": i})
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            #ema.update(ddpm)

        # Validation phase (less frequent, only every 5 epochs to save time)
        if i % 5 == 0 or i == epochs - 1:
            rn_rn_model.eval()
            print(f"🔍 Running validation for epoch {i}...")
            val_metrics = validate_diffusion_model(ddpm, val_loader, device, num_samples=200, num_predictions=5)
            
            # 2) NEW: noise-MSE on val (same objective used in training)
            val_noise_mse = compute_val_noise_mse(ddpm, val_loader, device, max_batches=None)

            # Log validation metrics to W&B
            val_metrics_log = {
                "val_rrmse": val_metrics['val_rrmse'],
                "val_mae": val_metrics['val_mae'],
                "val_loss": val_noise_mse,
                "epoch": i,
                "samples_evaluated": val_metrics['samples_evaluated'],
                "predictions_per_sample": val_metrics['predictions_per_sample']
            }
            run.log(val_metrics_log)
            
            print(f"📊 Epoch {i} | Loss: {loss_ema:.4f} | Val RRMSE: {val_metrics['val_rrmse']:.4f} | Val MAE: {val_metrics['val_mae']:.4f}")
        else:
            print(f"📊 Epoch {i} | Loss: {loss_ema:.4f}")

        # Generate and save sample images (every few epochs to save time)
        if i % 10 == 0 or i == epochs - 1:
            ddpm.eval()
            rn_rn_model.eval()
            with torch.no_grad():
                # Get conditioning samples from validation set
                sample_batch = next(iter(val_loader))
                cond_sample = sample_batch[0].to(device)[:16]  # Conditioning data
                target_sample = sample_batch[1][:16]  # Target shape reference
                if isinstance(ddpm, DDIM):
                    xh = ddpm.sample(16, (target_sample.shape[1], target_sample.shape[2], target_sample.shape[3]), device, cond=cond_sample, eta=0.0, ddim_steps=25)
                else:
                    xh = ddpm.sample(16, (target_sample.shape[1], target_sample.shape[2], target_sample.shape[3]), device, cond=cond_sample)
                for ch in range(xh.shape[1]):
                    channel_images = xh[:, ch:ch+1, :, :]  # (B, 1, H, W)
                    grid = torchvision.utils.make_grid(channel_images, nrow=4, normalize=True)
                    save_path = os.path.join(model_save_path, f"ddpm_sample_epoch_{i}_channel_{ch}.png")
                    torchvision.utils.save_image(grid, save_path)
                print(f"🖼️ Sample images saved for epoch {i}")

        # Save checkpoint
        checkpoint_data = {
            'epoch': i,
            'model_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_ema
        }
        
        # Add validation metrics if available
        if i % 5 == 0 or i == epochs - 1:
            checkpoint_data['val_rrmse'] = val_metrics['val_rrmse']
            checkpoint_data['val_mae'] = val_metrics['val_mae']
            checkpoint_data['val_noise_mse'] = val_noise_mse
            
        torch.save(checkpoint_data, os.path.join(model_save_path, f"ddpm_checkpoint_epoch_{i}.pt"))
        print(f"💾 Checkpoint saved for epoch {i}")
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"📈 Learning rate updated to: {current_lr:.2e}")
        run.log({"learning_rate": current_lr})

    print("✅ Training loop completed for diffusion model.")
    
    # Run final validation
    print("🔍 Running final validation...")
    final_val_metrics = validate_diffusion_model(ddpm, val_loader, device, num_samples=500, num_predictions=5)
    final_val_noise_mse = compute_val_noise_mse(ddpm, val_loader, device, max_batches=None)

    # Save the final model with the generated name
    final_model_path = f"models/{model_name}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'model_name': model_name,
        'model_slug': model_slug,
        'config': config,
        'final_epoch': epochs,
        'final_loss': loss_ema,
        'final_val_rrmse': final_val_metrics['val_rrmse'],
        'final_val_mae': final_val_metrics['val_mae']
    }, final_model_path)
    
    print(f"🎉 Model saved as {final_model_path}")
    print(f"📊 Final metrics - Loss: {loss_ema:.4f} | RRMSE: {final_val_metrics['val_rrmse']:.4f} | MAE: {final_val_metrics['val_mae']:.4f}")
    
    # Log the final model path and metrics to W&B
    run.log({
        "final_model_path": final_model_path,
        "final_val_rrmse": final_val_metrics['val_rrmse'],
        "final_val_mae": final_val_metrics['val_mae'],
        "final_val_noise_mse": final_val_noise_mse,  # NEW
        "training_completed": True
    })
    
    run.finish()
    return ddpm
            