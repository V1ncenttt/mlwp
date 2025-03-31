# 🌦️ Deterministic vs Probabilistic Machine Learning for Weather Prediction

This repository contains the codebase and experiments for my MSc thesis at Imperial College London. The goal is to compare **deterministic** machine learning models (e.g., CNNs, Transformers) with **probabilistic** generative models (e.g., diffusion models) for weather forecasting, using the [WeatherBench2](https://github.com/weatherbench2) dataset.

---

## 📌 Project Objectives

- **Prediction Accuracy:** How do deterministic and generative models compare in their forecasting performance?
- **Robustness to Noise:** Which model types better handle realistic noisy inputs?
- **Data Sparsity:** Can models maintain performance with sparse observational data?

---

## 🧠 Model Types

- **Deterministic Models**
  - ResNet-based CNNs
  - Vision Transformers (e.g., ViT, Senseiver-style)
  - LSTM (DSOVT-style)

- **Probabilistic Models**
  - Score-based diffusion models (e.g., SDD, SGD)
  - Spatial-aware diffusion with Voronoi encoding

---

## 🧪 Dataset

- **WeatherBench2** — A modern benchmark for global data-driven weather prediction.
- Fields: 2m temperature, 10m wind, mean sea level pressure, etc.
- Resolutions: 1.4°, 0.25°, 6-hourly timesteps

---

## 🧱 Structure
```bash
.
├── models/              # Model definitions (CNN, ViT, Diffusion, etc.)
├── data/                # WeatherBench2 loading and preprocessing
├── scripts/             # Training and evaluation scripts
├── configs/             # YAML config files
├── notebooks/           # Exploratory analysis and results
└── results/             # Logs, metrics, figures
```

---

## 📊 Evaluation Metrics

- RMSE, MAE, ACC, SSIM
- Skill score vs climatology
- Calibration plots for probabilistic outputs

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
