# 📂 data/

This folder contains the preprocessed WeatherBench2 subset used in this project.

⚠️ This directory is excluded from version control via `.gitignore`.

---

## 📥 How to Download the Dataset

To download and extract the necessary variables from the WeatherBench2 ERA5 dataset:
1.	Run the download script from the project root:
```bash
python scripts/download_data.py
```
This will:
	•	Open the public WeatherBench2 dataset stored on Google Cloud
	•	Extract 5 key surface-level variables (temperature, wind, pressure, moisture)
	•	Restrict to the 2019–2020 time range
	•	Save the result to data/

# ℹ️  Notes
	•	The full WeatherBench2 dataset is ~90TB. This script only downloads a small subset (~a few GB).
	•	You can adjust the script to change variables or time ranges as needed.


