# 🌦️ Machine Learning for Weather Field Reconstruction

This repository contains the codebase and experiments for my MSc thesis at Imperial College London. The focus is on **field reconstruction** - recovering complete spatial weather fields from sparse sensor observations using machine learning, comparing deterministic and generative approaches on the [WeatherBench2](https://github.com/weatherbench2) dataset.

---

## 📌 Project Objectives

- **Field Reconstruction Accuracy:** How well can models reconstruct full weather fields from sparse, irregular sensor data?
- **Conditioning Methods:** Which conditioning approaches (spatial, FiLM, hybrid) work best for diffusion models?
- **Model Comparison:** How do deterministic CNN-based models compare to generative diffusion models?

---

## 🧠 Implemented Models

### Deterministic Models
- **Fukami-style CNN** - Deep CNN with Voronoi tessellation preprocessing (adapted from [Fukami et al. 2021](https://www.nature.com/articles/s42256-021-00402-2))
- **Voronoi-Tesselation UNet** - UNet architecture with voronoï tesselation preprocessing
- **Voronoï-Tesselation ResNet** - Residual blocks for improved training stability, with voronoï tesselation preprocessing
- **Variational Autoencoder (VAE)** - Latent space reconstruction approach

### Generative Models
- **Conditional DDPM/DDIM** - Denoising diffusion models with multiple conditioning methods:
  - **Spatial Conditioning** - Direct concatenation of sensor mask and sparse field
  - **FiLM Conditioning** - Feature-wise Linear Modulation 
  - **Hybrid Conditioning** - Novel approach combining patchification, transformers, and cross-attention (inspired by [Zhuang et al. 2024](https://doi.org/10.1029/2024MS004395))
- **Conditional Wasserstein GAN (CWGAN)** - Adversarial approach for field reconstruction

### Baselines
- **Linear/Cubic Interpolation** - Classical scipy-based interpolation methods
- **Kriging** - Classical geostatistical method

---

## 🧪 Dataset

- **WeatherBench2** - Modern benchmark for global data-driven weather prediction
- **Variables**: 2m temperature, 10m wind components (u, v), mean sea level pressure, total column water vapor
- **Resolution**: 1.4° (64×32 grid), 6-hourly timesteps
- **Task**: Reconstruct full fields from sparse sensor observations (simulated from regular grid)

---

## 🏗️ Project Structure
```bash
mlwp/
├── data/                           # WeatherBench2 data and preprocessing
│   ├── README.md
│   ├── weatherbench2_5vars_3d.nc  # Raw dataset (5 variables, 3D)
│   └── weatherbench2_5vars_flat.nc # Preprocessed flat fields
├── download_data.py                # Data download script
├── src/
│   ├── field_reconstruction/       # Main field reconstruction experiments
│   │   ├── models/                 # Model implementations
│   │   │   ├── diffusion/          # DDPM/DDIM with conditioning
│   │   │   │   ├── ddpm.py        # Core diffusion implementation
│   │   │   │   ├── diffusion_unet.py # UNet with spatial/FiLM/hybrid conditioning
│   │   │   │   └── noise_schedule.py
│   │   │   ├── fukami.py          # CNN with Voronoi preprocessing
│   │   │   ├── vae.py             # Variational autoencoder
│   │   │   ├── cwgan.py           # Conditional Wasserstein GAN
│   │   │   └── baseline.py        # Interpolation baselines
│   │   ├── train.py               # Training script with multi-LR optimization
│   │   ├── test.py                # Evaluation with FLOP counting
│   │   ├── main.py                # Experiment orchestration
│   │   ├── config.yaml            # Configuration file
│   │   ├── utils.py               # Dataset loading and utilities
│   │   ├── voronoi.py             # Voronoi tessellation preprocessing
│   │   └── cluster_job.sh         # HPC cluster job script
│   └── forecast/                   # Future: multi-step forecasting
├── plots/                          # Generated visualizations
└── logs/                           # Training logs and outputs

---

## 📊 Evaluation Metrics

- **MSE, RMSE, MAE** - Standard reconstruction error metrics
- **SSIM** - Structural similarity for spatial coherence
- **Computational Cost** - FLOPs analysis for model efficiency
- **Visual Assessment** - Qualitative comparison of reconstructed fields

---

## 🔬 Conditioning Methods for Diffusion Models

A key contribution is the systematic comparison of conditioning approaches for weather field reconstruction:

### Spatial Conditioning (Baseline)
- Direct concatenation of sparse field + sensor mask with noise input
- Simple but effective approach

### FiLM Conditioning (Feature-wise Linear Modulation)
- Applies learned scale/shift parameters to UNet features
- Better separation of content and conditioning information

### Hybrid Conditioning (Novel)
- **Patchification**: Converts conditioning field to 8×8 patches
- **FiLM**: Applies feature-wise modulation to patch embeddings  
- **Positional Encoding**: Sinusoidal position embeddings for spatial awareness
- **Transformers**: Self-attention over patch sequences
- **Cross-Attention**: Attends UNet features to processed patches
- **Shared Architecture**: Single encoder with level-specific projections for efficiency

### Key Implementation Details
- **FLOP Analysis**: Computational cost comparison using fvcore library

---

## � Quick Start

### Setup
```bash
# Clone repository
git clone https://github.com/V1ncenttt/mlwp.git
cd mlwp

# Install dependencies
pip install -r src/field_reconstruction/requirements.txt

# Download WeatherBench2 subset
python download_data.py
```

### Training Models
```bash
cd src/field_reconstruction

# Train diffusion model with hybrid conditioning
python train.py --model ddpm --conditioning hybrid --epochs 100

# Train CNN baseline  
python train.py --model fukami --epochs 50

# Train VAE
python train.py --model vae --latent_dim 128 --epochs 75
```

### Evaluation
```bash
# Test trained models with FLOP counting
python test.py --model ddpm --conditioning hybrid --k 5

# Compare multiple models
python test.py --compare --models ddpm,fukami,vae
```

### Configuration
Edit `config.yaml` to modify:
- Model architectures and hyperparameters  
- Dataset parameters (variables, resolution, sparsity)
- Training settings (batch size, learning rates, schedulers)

---

## � Key References

- **WeatherBench2**: [Rasp et al. (2024)](https://doi.org/10.1029/2023MS004019) - Modern weather prediction benchmark
- **Hybrid Conditioning**: [Zhuang et al. (2024)](https://doi.org/10.1029/2024MS004395) - Generative diffusion for surrogate modeling  
- **Voronoi CNN**: [Fukami et al. (2021)](https://www.nature.com/articles/s42256-021-00402-2) - CNN-based field reconstruction
- **DDPM**: [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) - Denoising diffusion probabilistic models
- **DDIM**: [Song et al. (2021)](https://arxiv.org/abs/2010.02502) - Deterministic sampling for diffusion models

---

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+  
- xarray, einops, scipy
- fvcore (for FLOP counting)
- wandb (for experiment tracking)
- See `src/field_reconstruction/requirements.txt` for complete list

---

## � Supervision & Collaboration

- **Supervisor:** Dr. Sibo Cheng (Imperial College London / Institut Polytechnique de Paris)
- **Institution:** Imperial College London, Department of Earth Science and Engineering
- **Collaboration:** Institut Polytechnique de Paris (France)

---

## 📬 Contact

For questions about the implementation or research collaboration:
- [vincent.lefeuve@imperial.ac.uk](mailto:vincent.lefeuve@imperial.ac.uk)
- GitHub: [@V1ncenttt](https://github.com/V1ncenttt)
