import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, color, img_as_bool
import pandas as pd

# === CONFIG ===
IMAGE_PATH = "../images/sem_image.tif"
MASK_PATH = "../images/Mask.tif"
HFW_UM = 138.0
THRESH = 0.235
MIN_PORE_SIZE = 30

# === LOAD AND PROCESS ===
image = io.imread(IMAGE_PATH, as_gray=True)
mask = img_as_bool(io.imread(MASK_PATH))
height, width = image.shape
scale_um_per_px = HFW_UM / width

# Apply Gaussian blur and segmentation
image = filters.gaussian(image, sigma=1)
binary = (image < THRESH) & mask
binary = morphology.remove_small_objects(binary, min_size=MIN_PORE_SIZE)

# Label each pore and compute properties
labeled = measure.label(binary)
props = measure.regionprops(labeled)
areas_um2 = [p.area * (scale_um_per_px**2) for p in props]
diam_um = [p.equivalent_diameter * scale_um_per_px for p in props]

# Color overlay by label
overlay = color.label2rgb(labeled, image=image, bg_label=0, alpha=0.5, kind='overlay')

# === PLOTTING ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Pore overlay (left)
ax1.imshow(overlay)
ax1.set_title("Labeled Pore Segmentation")
ax1.axis("off")

# Histogram (right)
ax2.hist(diam_um, bins=20, color='skyblue', edgecolor='black')
ax2.set_xlabel("Equivalent Pore Diameter (µm)")
ax2.set_ylabel("Count")
ax2.set_title("Pore Size Distribution")
ax2.grid(True)

plt.tight_layout()
plt.savefig("../results/overlay_histogram_summary.png", dpi=300)
plt.show()

# Save data (optional)
df = pd.DataFrame({
    "Pore Area (µm²)": areas_um2,
    "Equivalent Diameter (µm)": diam_um
})
df.to_csv("../results/pore_stats_colored.csv", index=False)
print("→ Saved overlay + histogram figure and CSV")

