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
from loss_functions import cwgan_discriminator_loss, cwgan_generator_loss, get_loss_function

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
    
    model = create_model(model_name, nb_channels=nb_channels)
    
    if "cwgan" in model_name.lower():
        print("⚠️  Detected CWGAN model — using train_cwgan() instead of train()")
        generator, discriminator = model
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        train_cwgan(generator, discriminator, data, model_save_path, config)
        return  # Exit normal train() after calling train_cwgan
    
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