# 🌦️ Comparative Study of Deterministic and Generative Surrogate Models for Numerical Weather Prediction

This repository contains the codebase and experiments for my MSc thesis at Imperial College London. The goal is to compare **deterministic** machine learning models (e.g., CNNs, Transformers) with **probabilistic** generative models (e.g., diffusion models) for weather forecasting, using the [WeatherBench2](https://github.com/weatherbench2) dataset.

---

## 📌 Project Objectives

- **Prediction Accuracy:** How do deterministic and generative models compare in their forecasting performance?
- **Robustness to Noise:** Which model types better handle realistic noisy inputs?
- **Data Sparsity:** Can models maintain performance with sparse observational data?

---

## 🧠 Model Types
### Field Reconstruction

- **Deterministic Models**
  - ResNet-based CNNs
  - U-net
  - Vision Transformers (e.g., ViT (VITAE-SL))
  - Simple CNNS (Voronoi-CNN)

- **Probabilistic Models**
  - Score-based diffusion models (e.g., SDD, SGD)
  - Spatial-aware diffusion with Voronoi encoding
  - CWGAN (Conditional Wasserstein GANs)

---

## 🧪 Dataset

- **WeatherBench2** — A modern benchmark for global data-driven weather prediction.
- Fields: 2m temperature, 10m wind, mean sea level pressure, etc.
- Resolutions: 1.4°, 0.25°, 6-hourly timesteps

---

## 🧱 Structure
```bash
.
├── data/                # WeatherBench2 loading and preprocessing
├── plots/            
└── src/             
```

---

## 📊 Evaluation Metrics

- RMSE, MAE, ACC, SSIM
- Skill score vs climatology
- Calibration plots for probabilistic outputs

---

## 🗺️ Field Reconstruction

Field reconstruction is a core task in data-driven weather modeling. It consists in recovering full spatial fields (e.g. temperature, wind) from sparse and irregular observations — as often encountered in real-world meteorological sensor networks.

In this project, we simulate sparse sensor measurements from the WeatherBench2 dataset and compare several methods for reconstructing the full field:

### 🔬 Methods Compared

| Method | Description |
|--------|-------------|
| **Fukami-style CNN** | A convolutional neural network trained on **Voronoi-tessellated inputs**, where each sensor region is filled using a Voronoi mask. This architecture is adapted from [Fukami et al.](https://www.nature.com/articles/s42256-021-00402-2). |
| **Variational Autoencoder (VAE)** | A generative model that learns a low-dimensional latent representation of fields, then performs reconstruction by optimizing in latent space given sparse observations. |
| **Linear & Cubic Interpolation** | Classical interpolation techniques using `scipy.griddata`, serving as baselines. These are purely geometric and don’t learn from data. |

Each method takes as input:
- A 2-channel tensor:
  - Channel 1: sparse input field (e.g. Voronoi-tessellated or interpolated)
  - Channel 2: binary sensor mask
- The goal is to reconstruct the full target field and minimize reconstruction error (e.g. MSE).

### 🧪 Evaluation

We evaluate the methods on random held-out samples using:
- Mean squared error (MSE)
- Visual comparisons (ground truth vs. reconstruction)
- Robustness under varying sparsity levels

You can find visualizations and plots in the `plots/` folder after training.

---

## 📚 References

- [WeatherBench2: Rasp et al. (2024)](https://doi.org/10.1029/2023MS004019)
- [Generative Diffusion for Surrogate Modeling: Finn et al. (2024)](https://doi.org/10.1029/2024MS004395)

---

## 👤 Supervision

- **Supervisor:** Dr. Sibo Cheng (Imperial College London/ Institut Polytechnique de Paris)
- **Collaboration:** Institut polytechnique de Paris (France)

---

## 📢 Publication & PhD Opportunity

This thesis is part of a broader research direction aimed at high-impact publication (ICLR, NeurIPS, JAMES, GMD) and may evolve into a PhD depending on outcomes and funding.

---

## 🛠 Requirements

- Python 3.10+
- PyTorch, xarray, einops, WandB
- See `requirements.txt` for full setup

---

## 📬 Contact

For questions or collaboration inquiries, feel free to reach out:
- [vincent.lefeuve@imperial.ac.uk](mailto:vincent.lefeuve@imperial.ac.uk)
