# LLZO SEM Pore Analyzer

This repo contains Python code to analyze pores in LLZO-based solid-state electrolyte SEM images using adjustable thresholding and real-time segmentation feedback.

## 📂 Directory Structure

- \`src/\` — Python scripts (main code: \`threshold_tuner.py\`)
- \`images/\` — Input SEM images
- \`results/\` — Output masks, CSVs, figures
- \`.venv/\` — Local virtual environment (ignored)

## 🚀 Quickstart

\`\`\`bash
git clone https://github.com/ogarcia2/sem_porosity.git
cd sem_porosity
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/threshold_tuner.py
\`\`\`

## 🧪 Features
- Gaussian smoothing
- Real-time threshold adjustment via matplotlib slider
- Pore segmentation and object filtering

## 🔬 Input
Drop SEM images in \`images/\` and edit \`IMAGE_PATH\` in the script if needed.

