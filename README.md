# LLZO SEM Pore Analyzer

This repository provides a calibrated, automated pipeline for pore segmentation and size distribution analysis from SEM images of LiCoO₂–LLZTO composite cathodes. It includes threshold tuning, physical unit conversion, and publication-ready figure generation.

---

## 🔍 Features

- ✅ Calibrated pixel-to-micron scaling (30 µm = 145.377 px → 0.2063 µm/px)
- ✅ Gaussian-filtered thresholding and mask application
- ✅ Pore size metrics: area, equivalent diameter, porosity
- ✅ Color-coded overlay + histogram with matched color mapping
- ✅ Figure output with scale bar, sorted pore sizes, and optional colormap control
- ✅ CSV export of all measurements

---

## 🧪 Example Output

- `results/overlay_histogram_colormapped.png`  
  → Side-by-side figure: segmentation overlay and sorted diameter histogram  
- `results/pore_stats_colormapped.csv`  
  → Full table of pore diameters and areas

---

## 🚀 Quickstart

```bash
# Clone and enter
git clone https://github.com/ogarcia2/sem_porosity.git
cd sem_porosity

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the main analysis
python src/overlay_histogram_colormapped.py

Structure:

sem_porosity/
├── images/           # Input SEM image + mask
├── results/          # Output figures + CSV
├── src/              # All Python scripts
│   └── overlay_histogram_colormapped.py
├── README.md
├── requirements.txt
└── .venv/            # Optional virtualenv

✏ Citation / Acknowledgment
Used for image analysis of LLZO–LCO composite cathode SEMs in [manuscript ref]. Developed by Oskar Garcia and Bo Wang’s group at LLNL/SFSU.

