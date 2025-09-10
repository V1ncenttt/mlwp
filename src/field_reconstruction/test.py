import torch
import os
import numpy as np
from tqdm import tqdm
from models import FukamiNet, ReconstructionVAE
from models.diffusion.diffusion_unet import SimpleUnet
from models.diffusion.ddpm import DDPM, DDIM
import matplotlib.pyplot as plt
from utils import get_device
from plots_creator import plot_random_reconstruction, plot_l2_error_distributions, plot_reconstruction_per_snapshot_per_field
from utils import create_model, get_device
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim
from pykrige.ok import OrdinaryKriging
from fvcore.nn import flop_count

from pathlib import Path  # <-- make sure this is imported

# Try to import ptflops with better error reporting
try:
    from ptflops import get_model_complexity_info
    ptflops_available = True
    print("✅ ptflops imported successfully")
except ImportError as e:
    print(f"⚠️ ptflops import failed: {e}")
    print("⚠️ Install with: pip install ptflops")
    ptflops_available = False
    get_model_complexity_info = None
except Exception as e:
    print(f"⚠️ ptflops unexpected error: {e}")
    ptflops_available = False
    get_model_complexity_info = None

def count_model_flops(model, sample_input, model_type="gan", k=5):
    """
    Count FLOPs for one prediction using both fvcore and ptflops for comparison.
    
    Args:
        model: The model to count FLOPs for
        sample_input: Sample input tensor
        model_type: Type of model ("gan", "diffusion", etc.)
        k: Number of ensemble predictions (for diffusion/gan models)
    
    Returns:
        dict: Dictionary with FLOP counts and analysis from both libraries
    """
    results = {
        "fvcore": {"available": flop_count is not None},
        "ptflops": {"available": ptflops_available},  # Use global variable
        "ensemble_size": k
    }
    
    print(f"\n🔬 Comparing FLOP counters for {model_type} model:")
    print(f"📋 fvcore available: {results['fvcore']['available']}")
    print(f"📋 ptflops available: {results['ptflops']['available']}")
    
    try:
        with torch.no_grad():
            if "diffusion" in model_type:
                # For diffusion models, count FLOPs for the sampling process
                batch_size, channels, H, W = sample_input.shape
                target_shape = (5, H, W)  # Assuming 5 output channels
                
                # Count FLOPs for one denoising step (approximation)
                noise_sample = torch.randn(1, *target_shape, device=sample_input.device)
                cond_sample = sample_input[:1]  # Use first sample as conditioning
                timestep = torch.randint(0, 1000, (1,), device=sample_input.device)
                
                # Method 1: fvcore
                if results["fvcore"]["available"]:
                    try:
                        flops_dict, _ = flop_count(
                            model.eps_model,  # The UNet inside DDIM
                            (noise_sample, timestep, cond_sample)
                        )
                        unet_gflops = sum(flops_dict.values())
                        unet_flops = unet_gflops * 1e9  # Convert GFLOPs to FLOPs
                        ddim_steps = 50
                        
                        results["fvcore"].update({
                            "single_step_gflops": unet_gflops,
                            "total_flops": unet_flops * ddim_steps * k,
                            "flops_per_prediction": unet_flops * ddim_steps
                        })
                        print(f"📊 fvcore: {unet_gflops:.3f} GFLOPs per UNet step")
                    except Exception as e:
                        print(f"❌ fvcore failed: {e}")
                        results["fvcore"]["error"] = str(e)
                
                # Method 2: ptflops (approximate for diffusion - harder to measure exact sampling)
                if results["ptflops"]["available"]:
                    try:
                        # Measure just the UNet model
                        input_shape = (target_shape[0] + cond_sample.shape[1], target_shape[1], target_shape[2])
                        dummy_input = torch.cat([noise_sample, cond_sample], dim=1)
                        
                        # Create a wrapper to handle the timestep input
                        class UNetWrapper(torch.nn.Module):
                            def __init__(self, unet):
                                super().__init__()
                                self.unet = unet
                            
                            def forward(self, x):
                                noise_part = x[:, :target_shape[0]]
                                cond_part = x[:, target_shape[0]:]
                                t = torch.randint(0, 1000, (x.shape[0],), device=x.device)
                                return self.unet(noise_part, t, cond_part)
                        
                        wrapper = UNetWrapper(model.eps_model)
                        macs, params = get_model_complexity_info(
                            wrapper, 
                            input_shape, 
                            print_per_layer_stat=False,
                            verbose=False
                        )
                        
                        # Handle case where macs might be a string  
                        if isinstance(macs, str):
                            print(f"⚠️ ptflops returned string MACs: {macs}")
                            import re
                            numbers = re.findall(r'[\d.]+', macs)
                            if numbers:
                                macs = float(numbers[0])
                            else:
                                raise ValueError(f"Cannot parse MACs from string: {macs}")
                        
                        ptflops_gflops = float(macs) / 1e9
                        ddim_steps = 50
                        
                        results["ptflops"].update({
                            "single_step_gflops": ptflops_gflops,
                            "total_flops": ptflops_gflops * 1e9 * ddim_steps * k,
                            "flops_per_prediction": ptflops_gflops * 1e9 * ddim_steps
                        })
                        print(f"📊 ptflops: {ptflops_gflops:.3f} GFLOPs per UNet step")
                    except Exception as e:
                        print(f"❌ ptflops failed: {e}")
                        results["ptflops"]["error"] = str(e)
                        
            elif "gan" in model_type:
                # For GAN models, count FLOPs for generator forward pass
                sample_for_flops = sample_input[:1]
                
                # Method 1: fvcore
                if results["fvcore"]["available"]:
                    try:
                        flops_dict, _ = flop_count(model, (sample_for_flops,))
                        gen_gflops = sum(flops_dict.values())
                        
                        results["fvcore"].update({
                            "single_forward_gflops": gen_gflops,
                            "total_flops": gen_gflops * 1e9 * k,
                            "flops_per_prediction": gen_gflops * 1e9
                        })
                        print(f"📊 fvcore: {gen_gflops:.3f} GFLOPs per forward pass")
                    except Exception as e:
                        print(f"❌ fvcore failed: {e}")
                        results["fvcore"]["error"] = str(e)
                
                # Method 2: ptflops
                if results["ptflops"]["available"]:
                    try:
                        input_shape = sample_for_flops.shape[1:]
                        macs, params = get_model_complexity_info(
                            model, 
                            input_shape, 
                            print_per_layer_stat=False,
                            verbose=False
                        )
                        
                        # Handle case where macs might be a string
                        if isinstance(macs, str):
                            print(f"⚠️ ptflops returned string MACs: {macs}")
                            import re
                            numbers = re.findall(r'[\d.]+', macs)
                            if numbers:
                                macs = float(numbers[0])
                            else:
                                raise ValueError(f"Cannot parse MACs from string: {macs}")
                        
                        ptflops_gflops = float(macs) / 1e9
                        
                        results["ptflops"].update({
                            "single_forward_gflops": ptflops_gflops,
                            "total_flops": ptflops_gflops * 1e9 * k,
                            "flops_per_prediction": ptflops_gflops * 1e9
                        })
                        print(f"📊 ptflops: {ptflops_gflops:.3f} GFLOPs per forward pass")
                    except Exception as e:
                        print(f"❌ ptflops failed: {e}")
                        results["ptflops"]["error"] = str(e)
                        
            else:
                # For other models (deterministic), count single forward pass
                sample_for_flops = sample_input[:1]
                k = 1  # No ensemble for deterministic models
                results["ensemble_size"] = k
                
                # Method 1: fvcore
                if results["fvcore"]["available"]:
                    try:
                        flops_dict, _ = flop_count(model, (sample_for_flops,))
                        fvcore_gflops = sum(flops_dict.values())
                        
                        results["fvcore"].update({
                            "single_forward_gflops": fvcore_gflops,
                            "total_flops": fvcore_gflops * 1e9,
                            "flops_per_prediction": fvcore_gflops * 1e9
                        })
                        print(f"📊 fvcore: {fvcore_gflops:.3f} GFLOPs per forward pass")
                    except Exception as e:
                        print(f"❌ fvcore failed: {e}")
                        results["fvcore"]["error"] = str(e)
                
                # Method 2: ptflops
                if results["ptflops"]["available"]:
                    try:
                        print(f"🔍 ptflops debug: input_shape = {sample_for_flops.shape[1:]}")
                        print(f"🔍 ptflops debug: model type = {type(model).__name__}")

                        input_shape = tuple(sample_for_flops.shape[1:])
                        macs, params = get_model_complexity_info(
                            model, 
                            input_shape, 
                            print_per_layer_stat=False,
                            verbose=False
                        )
                        
                        print(f"🔍 ptflops debug: macs = {macs} (type: {type(macs)})")
                        print(f"🔍 ptflops debug: params = {params} (type: {type(params)})")
                        
                        # Handle case where macs might be a string
                        if isinstance(macs, str):
                            print(f"⚠️ ptflops returned string MACs: {macs}")
                            # Try to extract number from string if possible
                            import re
                            numbers = re.findall(r'[\d.]+', macs)
                            if numbers:
                                macs = float(numbers[0])
                                print(f"🔍 Extracted MACs as float: {macs}")
                            else:
                                raise ValueError(f"Cannot parse MACs from string: {macs}")
                        
                        ptflops_gflops = float(macs) / 1e9
                        
                        results["ptflops"].update({
                            "single_forward_gflops": ptflops_gflops,
                            "total_flops": ptflops_gflops * 1e9,
                            "flops_per_prediction": ptflops_gflops * 1e9
                        })
                        print(f"📊 ptflops: {ptflops_gflops:.3f} GFLOPs per forward pass")
                    except Exception as e:
                        print(f"❌ ptflops failed: {type(e).__name__}: {e}")
                        import traceback
                        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
                        results["ptflops"]["error"] = str(e)
        
        # Print comparison summary
        print(f"\n📋 FLOP Comparison Summary:")
        if "single_forward_gflops" in results.get("fvcore", {}) and "single_forward_gflops" in results.get("ptflops", {}):
            fv_flops = results["fvcore"]["single_forward_gflops"]
            pt_flops = results["ptflops"]["single_forward_gflops"]
        # Print comparison summary
        print(f"\n📋 FLOP Comparison Summary:")
        
        # Check for different types of FLOP measurements
        fv_key = None
        pt_key = None
        
        if "single_forward_gflops" in results.get("fvcore", {}):
            fv_key = "single_forward_gflops"
        elif "single_step_gflops" in results.get("fvcore", {}):
            fv_key = "single_step_gflops"
            
        if "single_forward_gflops" in results.get("ptflops", {}):
            pt_key = "single_forward_gflops"
        elif "single_step_gflops" in results.get("ptflops", {}):
            pt_key = "single_step_gflops"
        
        if fv_key and pt_key:
            fv_flops = results["fvcore"][fv_key]
            pt_flops = results["ptflops"][pt_key]
            ratio = pt_flops / fv_flops if fv_flops > 0 else float('inf')
            print(f"   • fvcore: {fv_flops:.3f} GFLOPs")
            print(f"   • ptflops: {pt_flops:.3f} GFLOPs")
            print(f"   • Ratio (ptflops/fvcore): {ratio:.2f}x")
        elif fv_key:
            print(f"   • fvcore: {results['fvcore'][fv_key]:.3f} GFLOPs")
            print(f"   • ptflops: Not available")
        elif pt_key:
            print(f"   • fvcore: Not available")
            print(f"   • ptflops: {results['ptflops'][pt_key]:.3f} GFLOPs")
        
        # For backward compatibility, return the fvcore results in the old format
        if "flops_per_prediction" in results.get("fvcore", {}):
            return {
                "total_flops": results["fvcore"]["total_flops"],
                "flops_per_prediction": "Error", 
                "ensemble_size": k,
                "comparison_results": results
            }
        
    except Exception as e:
        print(f"⚠️ Error counting FLOPs: {e}")
        return {"total_flops": "Error", "flops_per_prediction": "Error", "comparison_results": results}

def kriging_interpolation(yx, values, H, W, model='exponential'):
    x = yx[:, 1].astype(np.float64)
    y = yx[:, 0].astype(np.float64)
    z = values.astype(np.float64)

    gridx = np.arange(W, dtype=float)
    gridy = np.arange(H, dtype=float)

    try:
        ok = OrdinaryKriging(x, y, z, variogram_model=model, verbose=False, enable_plotting=False)
        interp, _ = ok.execute('grid', gridx, gridy)
    except Exception as e:
        print(f"Kriging failed: {e}")
        interp = np.zeros((H, W))  # fallback

    return interp

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

def evaluate_interp(test_loader, mode: str = 'cubic', nb_channels: int = 2, variable_names=None):
    device = get_device()

    print(f"✅ Evaluating model on {len(test_loader.dataset)} samples")
    print(f"✅ Model type: Interpolation ({mode})")
    print(f"✅ Device: {device}")
    
    rrmse_total = []
    mae_total = []
    ssim_total = []
    l2_errors = [[] for _ in range(nb_channels)]
    n = 0
    num_vars = None

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        
        for inputs, targets in pbar:
            
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_np = inputs.cpu().numpy()   # (B, 1 + Nvars, H, W)
            targets_np = targets.cpu().numpy() # (B, Nvars, H, W)

            B, C, H, W = inputs_np.shape
            num_vars = C - 1  # First channel is the shared mask

            preds = []

            for i in range(B):
                mask = inputs_np[i, 0, :, :]  # shape: (H, W)
                yx = np.argwhere(mask > 0)

                pred_sample = []
                for v in range(num_vars):
                    voronoi = inputs_np[i, v + 1, :, :]  # v-th variable's tessellated field
                    values = voronoi[mask > 0]

                    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                    if mode in ['linear', 'cubic']:
                        interp = griddata(yx, values, (grid_y, grid_x), method=mode, fill_value=0.0)
                    elif mode == "kriging":
                        interp = kriging_interpolation(yx, values, H, W)
                    pred_sample.append(torch.tensor(interp, dtype=torch.float32))

                preds.append(torch.stack(pred_sample))  # (Nvars, H, W)

            preds = torch.stack(preds).to(device)  # (B, Nvars, H, W)

            rrmse_batch = []
            mae_batch = []
            ssim_batch = []

            for v in range(num_vars):
                pred_v = preds[:, v, :, :]
                target_v = targets[:, v, :, :]

                rrmse_v = rrmse(pred_v, target_v).item()
                mae_v = mae(pred_v, target_v).item()
                ssim_v = np.mean([
                    ssim(
                        pred_v[i].cpu().numpy().astype(np.float32),
                        target_v[i].cpu().numpy().astype(np.float32),
                        data_range=(target_v[i].max() - target_v[i].min() + 1e-8).item()
                    )
                    for i in range(pred_v.shape[0])
                ])

                l2_v = ((pred_v - target_v) ** 2).mean(dim=(1, 2))
                l2_errors[v].extend(l2_v.tolist())
                
                rrmse_batch.append(rrmse_v)
                mae_batch.append(mae_v)
                ssim_batch.append(ssim_v)
                

            rrmse_total.append(np.array(rrmse_batch) * B)
            mae_total.append(np.array(mae_batch) * B)
            ssim_total.append(np.array(ssim_batch) * B)
            n += B

            pbar.set_postfix({
                "RRMSE": np.mean(rrmse_batch),
                "MAE": np.mean(mae_batch),
            })

    rrmse_total = np.sum(rrmse_total, axis=0) / n
    mae_total = np.sum(mae_total, axis=0) / n
    ssim_total = np.sum(ssim_total, axis=0) / n

    if variable_names is None:
        variable_names = [f"Var{i}" for i in range(len(rrmse_total))]

    print("📈 Per-variable Metrics:")
    for idx, var in enumerate(variable_names):
        print(f"✅ {var}: RRMSE={rrmse_total[idx]:.4f}, MAE={mae_total[idx]:.4f}, SSIM={ssim_total[idx]:.4f}")

    print("\n📊 Overall Averages:")
    print(f"✅ Avg RRMSE={np.mean(rrmse_total):.4f}, Avg MAE={np.mean(mae_total):.4f}, Avg SSIM={np.mean(ssim_total):.4f}")

    # Convert to dict
    l2_errors_dict = {var: l2_errors[i] for i, var in enumerate(variable_names)}

    # 📊 Plot L2 Error Distributions
    save_dir = Path("plots/evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    indices = [3, 17, 42, 88, 123]
    if mode == "kriging":
        plot_l2_error_distributions(l2_errors_dict, variable_names, "Kriging", str(save_dir),)
        plot_random_reconstruction(
            model="kriging",
            val_loader=test_loader,
            device=device,
            model_name="kriging",
            save_dir=str(save_dir),
            num_samples=7
        )
        
        plot_reconstruction_per_snapshot_per_field(
            model="kriging",
            val_loader=test_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_name="kriging",
            save_dir="eval_plots",
            indices=indices,
        )
        
    elif mode == "cubic":
        plot_l2_error_distributions(l2_errors_dict, variable_names, "Cubic_Interpolation", str(save_dir),)
        plot_random_reconstruction(
            model="cubic_interpolation",
            val_loader=test_loader,
            device=device,
            model_name="cubic_interpolation",
            save_dir=str(save_dir),
            num_samples=7
        )
        
        plot_reconstruction_per_snapshot_per_field(
            model="cubic_interpolation",
            val_loader=test_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_name="cubic_interpolation",
            save_dir="eval_plots",
            indices=indices,
        )
        

    return {
        "rrmse": np.mean(rrmse_total),
        "mae": np.mean(mae_total),
        "ssim": np.mean(ssim_total),
        "rrmse_per_var": rrmse_total,
        "mae_per_var": mae_total,
        "ssim_per_var": ssim_total,
        "l2_per_var": l2_errors,
    }

def evaluate(model_type, test_loader, checkpoint_path, variable_names=None, config_file=None):
    
    device = get_device()
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    nb_channels = sample_input.shape[1]
    print(f'model_type: {model_type}')

    if model_type == "cubic_interpolation":
        return evaluate_interp(test_loader, mode="cubic", nb_channels=nb_channels, variable_names=variable_names)
        
    elif model_type == "kriging":
        return evaluate_interp(test_loader, mode="kriging", nb_channels=nb_channels, variable_names=variable_names)
        
    elif "gan" in model_type:
        print("🟢 Evaluating GAN with random noise injection (1st layer)")
        return evaluate_ensemble_model(
            model_type=model_type,
            test_loader=test_loader,
            checkpoint_path=checkpoint_path,
            variable_names=variable_names,
            config_file=config_file
        )
        
    elif "diffusion" in model_type:
        print("🟢 Evaluating Diffusion model")
        print("k=1")
        return evaluate_ensemble_model(
            model_type=model_type,
            test_loader=test_loader,
            checkpoint_path=checkpoint_path,
            variable_names=variable_names,
            config_file=config_file,
            k=5
        )
        
       

    model = create_model(model_type, nb_channels=nb_channels)
    if "gan" in model_type:
        model = model[0]
    # Insert channel check before loading state dict
    print(f"🟢 Generator input channels: {nb_channels}")
    
    # Check output channels based on model type
    if hasattr(model, 'final') and hasattr(model.final, 'out_channels'):
        output_channels = model.final.out_channels
    elif hasattr(model, 'out_channels'):
        output_channels = model.out_channels
    elif isinstance(model, torch.nn.Sequential) and len(model) > 0:
        last_layer = model[-1]
        if hasattr(last_layer, 'out_channels'):
            output_channels = last_layer.out_channels
        else:
            output_channels = 'Unknown'
    else:
        output_channels = 'Unknown'
    
    print(f"🟢 Generator output channels: {output_channels}")
    model = model.to(device)
    checkpoint_path = os.path.join("models/saves", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"✅ Evaluating model on {len(test_loader.dataset)} samples")
    print(f"✅ Model type: {model_type}")
    print(f"✅ Device: {device}")

    # Count FLOPs for one prediction
    print("🧮 Counting FLOPs...")
    flop_results = count_model_flops(model, sample_input, model_type, k=1)
    if (flop_results["total_flops"] != "N/A (fvcore not installed)" and 
        flop_results["total_flops"] != "Error" and 
        isinstance(flop_results["total_flops"], (int, float)) and
        isinstance(flop_results["flops_per_prediction"], (int, float))):
        print(f"📊 FLOPs per prediction: {flop_results['flops_per_prediction']:,}")
        print(f"📊 Total FLOPs (single prediction): {flop_results['total_flops']:,}")
    else:
        print(f"⚠️ FLOP counting not available: {flop_results['total_flops']}")

    rrmse_total, mae_total, ssim_total = [], [], []
    l2_errors = [[] for _ in range(nb_channels)]
    n_total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        
        for inputs, targets in pbar:
            
            
            inputs, targets = inputs.to(device), targets.to(device)

            if model_type == "vae":
                recon_x, mu, logvar = model(inputs)
                preds = recon_x
            elif "vitae" in model_type:
                pred_dec, pred_enc = model(inputs)
                preds = pred_dec
            elif "gan" in model_type and 'gan_randomness' in config_file and config_file['gan_randomness'] == "True":
                injection_mode = config_file['gan_injection_mode']
                pass
            else:
                if "gan" in model_type: 
                    inputs = inputs[:, 1:, :, :]  # Skip the first channel if it's a mask
                preds = model(inputs)

            
                
            batch_size, nb_channels, H, W = targets.shape
            rrmse_batch, mae_batch, ssim_batch = [], [], []

            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            for v in range(nb_channels):
                preds_v = preds_np[:, v, :, :]
                targets_v = targets_np[:, v, :, :]

                rrmse_val = np.sqrt(np.mean((preds_v - targets_v) ** 2)) / (np.sqrt(np.mean(targets_v ** 2)) + 1e-8)
                mae_val = np.mean(np.abs(preds_v - targets_v))
                ssim_val = np.mean([
                    ssim(preds_v[i], targets_v[i], data_range=targets_v[i].max() - targets_v[i].min() + 1e-8)
                    for i in range(preds_v.shape[0])
                ])
                l2_vals = np.mean((preds_v - targets_v) ** 2, axis=(1, 2))
                l2_errors[v].extend(l2_vals.tolist())

                rrmse_batch.append(rrmse_val)
                mae_batch.append(mae_val)
                ssim_batch.append(ssim_val)

            rrmse_total.append(np.array(rrmse_batch) * batch_size)
            mae_total.append(np.array(mae_batch) * batch_size)
            ssim_total.append(np.array(ssim_batch) * batch_size)
            n_total += batch_size

            pbar.set_postfix({
                "RRMSE": np.mean(rrmse_batch),
                "MAE": np.mean(mae_batch),
                "SSIM": np.mean(ssim_batch)
            })

    rrmse_total = np.sum(rrmse_total, axis=0) / n_total
    mae_total = np.sum(mae_total, axis=0) / n_total
    ssim_total = np.sum(ssim_total, axis=0) / n_total

    if variable_names is None:
        variable_names = [f"Var{i}" for i in range(len(rrmse_total))]

    print("📈 Per-variable Metrics:")
    for idx, var in enumerate(variable_names):
        print(f"✅ {var}: RRMSE={rrmse_total[idx]:.4f}, MAE={mae_total[idx]:.4f}, SSIM={ssim_total[idx]:.4f}")

    print("\n📊 Overall Averages:")
    print(f"✅ Avg RRMSE={np.mean(rrmse_total):.4f}, Avg MAE={np.mean(mae_total):.4f}, Avg SSIM={np.mean(ssim_total):.4f}")

    # Convert to dict
    l2_errors_dict = {var: l2_errors[i] for i, var in enumerate(variable_names)}

    # 📊 Plot L2 Error Distributions
    save_dir = Path("plots/evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plot_l2_error_distributions(l2_errors_dict, variable_names, model_type, str(save_dir),)
    
    plot_random_reconstruction(
        model=model,
        val_loader=test_loader,
        device=device,
        model_name=model_type,
        save_dir=str(save_dir),
        num_samples
        =7
    )
    
    indices = [3, 17, 42, 88, 123]

    plot_reconstruction_per_snapshot_per_field(
        model=model,
        val_loader=test_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_name=model_type,
        save_dir="eval_plots",
        indices=indices,
    )
    
    return {
        "rrmse": np.mean(rrmse_total),
        "mae": np.mean(mae_total),
        "ssim": np.mean(ssim_total),
        "rrmse_per_var": rrmse_total,
        "mae_per_var": mae_total,
        "ssim_per_var": ssim_total,
        "l2_per_var": l2_errors,
        "flop_results": flop_results,
    }

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
    

def evaluate_ensemble_model(model_type, test_loader, checkpoint_path, variable_names=None, config_file=None, k=25):
    """
    Evaluate an ensemble model by averaging multiple predictions
    """
    
    device = get_device()
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input.to(device)
    nb_channels = sample_input.shape[1] -1

    
    
    if "gan" in model_type:
        model = create_model(model_type, nb_channels=nb_channels+1)
        model = model[0].to(device)
        print(f"🟢 Generator input channels: {nb_channels+1}")
        print(f"🟢 Generator output channels: {model.final.out_channels if hasattr(model, 'final') else 'Unknown'}")
    elif "diffusion" in model_type:
        model = create_model(model_type, nb_channels=nb_channels+1)
        T = 1000  # Number of diffusion steps, can be adjusted
        rn_rn_model = model.to(device)
        ddpm = DDIM(rn_rn_model, (1e-4, 0.02), T).to(device)
        model = ddpm
        
    checkpoint_path = os.path.join("models/saves", checkpoint_path)
    
    if "diffusion" in model_type:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    model.eval()
    
    print(f"✅ Loaded model from {checkpoint_path}")
    print(f"✅ Evaluating model on {len(test_loader.dataset)} samples")
    print(f"✅ Model type: {model_type}")
    print(f"✅ Device: {device}")
    
    # Count FLOPs for ensemble predictions (k=5 by default)
    print("🧮 Counting FLOPs for ensemble predictions...")
    flop_results = count_model_flops(model, sample_input, model_type, k=k)
    if (flop_results["total_flops"] != "N/A (fvcore not installed)" and 
        flop_results["total_flops"] != "Error" and 
        isinstance(flop_results["total_flops"], (int, float)) and
        isinstance(flop_results["flops_per_prediction"], (int, float))):
        print(f"📊 FLOPs per single prediction: {flop_results['flops_per_prediction']:,}")
        print(f"📊 Total FLOPs (ensemble of {k}): {flop_results['total_flops']:,}")
        if "diffusion" in model_type:
            print(f"📊 Note: Diffusion FLOPs include {50} DDIM sampling steps")
    else:
        print(f"⚠️ FLOP counting not available: {flop_results['total_flops']}")
    
    # Limit to 200 samples for diffusion models to speed up evaluation
    max_samples = 1000 if "diffusion" in model_type else len(test_loader.dataset)
    if "diffusion" in model_type:
        print(f"⚡ Fast evaluation mode: limiting to {max_samples} samples for diffusion model")
        if isinstance(model, DDIM):
            print("⚠️ Using DDIM for sampling, ensure model is compatible with this method")
            pass #Might set eta and t in the future
    
    rrmse_total, mae_total, ssim_total = [], [], []
    l2_errors = [[] for _ in range(nb_channels)]
    n_total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Break early if we've processed enough samples for diffusion models
            if "diffusion" in model_type and n_total >= max_samples:
                print(f"⚡ Reached {max_samples} samples limit for diffusion model evaluation")
                break
            print(f"Batch {batch_idx + 1}/{len(test_loader)}")
            
            inputs, targets = inputs.to(device), targets.to(device)
            all_preds = []

            for _ in range(k):
                # Prepare random input for each ensemble member
                if "gan" in model_type:
                    inputs_mod = inputs[:, 1:, :, :]  # Skip the first channel if it's a mask
                    z_random = torch.randn_like(inputs_mod[:, 0:1, :, :])  # shape: (B, 1, H, W)
                    inputs_random = torch.cat([inputs_mod, z_random], dim=1)
                    preds = model(inputs_random)
                    #Print the shape of the predictions
                    #print(f"Predictions shape: {preds.shape}")
                elif "diffusion" in model_type:
                    # Use the ddpm model to generate predictions. dont skip the first channel. to get the predictions we need sample()
                    cond = inputs  # Use full inputs as conditioning (including mask)
                    batch_size, _, H, W = targets.shape
                    cfgscale = 1.5
                    print(cfgscale)
                    # Generate samples using the DDPM
                    preds = model.sample(
                        n_sample=batch_size,
                        size=(nb_channels, H, W),  # Size of the target tensor
                        device=device,
                        cond=cond,  # Pass conditioning tensor
                        ddim_steps=100,
                        cfg_scale=cfgscale,
                        noise_aware_cfg = False
                    )
                else:
                    preds = model(inputs)
                all_preds.append(preds.cpu().numpy())

            all_preds = np.array(all_preds)  # shape: (k, B, C, H, W)
            all_preds_mean = np.mean(all_preds, axis=0)  # shape: (B, C, H, W)
            all_preds_var = np.var(all_preds, axis=0)

            batch_size, nb_channels, H, W = targets.shape
            rrmse_batch, mae_batch, ssim_batch = [], [], []

            preds_np = all_preds_mean
            targets_np = targets.detach().cpu().numpy()

            # If model_type is GAN and input channel was reduced, output nb_channels may be less by 1
            eval_channels = nb_channels
            for v in range(eval_channels):
                preds_v = preds_np[:, v, :, :]
                targets_v = targets_np[:, v, :, :]

                rrmse_val = np.sqrt(np.mean((preds_v - targets_v) ** 2)) / (np.sqrt(np.mean(targets_v ** 2)) + 1e-8)
                mae_val = np.mean(np.abs(preds_v - targets_v))
                ssim_val = np.mean([
                    ssim(preds_v[i], targets_v[i], data_range=targets_v[i].max() - targets_v[i].min() + 1e-8)
                    for i in range(preds_v.shape[0])
                ])
                l2_vals = np.mean((preds_v - targets_v) ** 2, axis=(1, 2))
                l2_errors[v].extend(l2_vals.tolist())

                rrmse_batch.append(rrmse_val)
                mae_batch.append(mae_val)
                ssim_batch.append(ssim_val)

            rrmse_total.append(np.array(rrmse_batch) * batch_size)
            mae_total.append(np.array(mae_batch) * batch_size)
            ssim_total.append(np.array(ssim_batch) * batch_size)
            n_total += batch_size

            pbar.set_postfix({
                "RRMSE": np.mean(rrmse_batch),
                "MAE": np.mean(mae_batch),
                "SSIM": np.mean(ssim_batch)
            })

    rrmse_total = np.sum(rrmse_total, axis=0) / n_total
    mae_total = np.sum(mae_total, axis=0) / n_total
    ssim_total = np.sum(ssim_total, axis=0) / n_total

    if variable_names is None:
        variable_names = [f"Var{i}" for i in range(len(rrmse_total))]

    print("📈 Per-variable Metrics:")
    for idx, var in enumerate(variable_names):
        print(f"✅ {var}: RRMSE={rrmse_total[idx]:.4f}, MAE={mae_total[idx]:.4f}, SSIM={ssim_total[idx]:.4f}")

    print("\n📊 Overall Averages:")
    print(f"✅ Avg RRMSE={np.mean(rrmse_total):.4f}, Avg MAE={np.mean(mae_total):.4f}, Avg SSIM={np.mean(ssim_total):.4f}")

    # Convert to dict
    l2_errors_dict = {var: l2_errors[i] for i, var in enumerate(variable_names)}

    # 📊 Plot L2 Error Distributions
    save_dir = Path("plots/evaluation")
    save_dir.mkdir(parents=True, exist_ok=True)

    #plot_l2_error_distributions(l2_errors_dict, variable_names, model_type, str(save_dir),)
    # TODO: Uncomment and/or add if statement
    """
    plot_random_reconstruction(
        model=model,
        val_loader=test_loader,
        device=device,
        model_name=model_type,
        save_dir=str(save_dir),
        num_samples=7
    )
    """
    indices = [3, 17, 42, 88, 123]
    print("-----")
    
    plot_reconstruction_per_snapshot_per_field(
        model=model,
        val_loader=test_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_name=model_type,
        save_dir="eval_plots",
        indices=indices,
    )
    
    
    return {
        "rrmse": np.mean(rrmse_total),
        "mae": np.mean(mae_total),
        "ssim": np.mean(ssim_total),
        "rrmse_per_var": rrmse_total,
        "mae_per_var": mae_total,
        "ssim_per_var": ssim_total,
        "l2_per_var": l2_errors,
        "flop_results": flop_results,
    }

                
                    