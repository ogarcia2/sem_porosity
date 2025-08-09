import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, img_as_bool

# === CONFIG ===
IMAGE_PATH = "../images/sem_image.tif"
MASK_PATH = "../images/Mask.tif"
HFW_UM = 138.0
THRESH = 0.235
MIN_PORE_SIZE = 30
EXPORT = True

# === LOAD IMAGES ===
image = io.imread(IMAGE_PATH, as_gray=True)
mask = img_as_bool(io.imread(MASK_PATH))
height, width = image.shape
scale_um_per_px = HFW_UM / width

# === PREPROCESS + SEGMENT ===
image = filters.gaussian(image, sigma=1)
segmented = (image < THRESH) & mask
segmented = morphology.remove_small_objects(segmented, min_size=MIN_PORE_SIZE)

# === ANALYZE ===
labeled = measure.label(segmented)
props = measure.regionprops(labeled)
areas_um2 = [p.area * (scale_um_per_px**2) for p in props]
diam_um = [p.equivalent_diameter * scale_um_per_px for p in props]
porosity = (np.sum(segmented) / np.sum(mask)) * 100

# === PRINT STATS ===
print("=== Pore Statistics ===")
print(f"Detected pores:       {len(areas_um2)}")
print(f"Mean area (µm²):      {np.mean(areas_um2):.2f}")
print(f"Median area (µm²):    {np.median(areas_um2):.2f}")
print(f"Mean diameter (µm):   {np.mean(diam_um):.2f}")
print(f"Porosity:             {porosity:.2f}%")

# === EXPORT CSV & FIGURE ===
if EXPORT:
    df = pd.DataFrame({
        "Pore Area (µm²)": areas_um2,
        "Equivalent Diameter (µm)": diam_um
    })
    df.to_csv("../results/pore_stats.csv", index=False)
    print("→ Saved CSV: ../results/pore_stats.csv")

    plt.figure(figsize=(8, 5))
    plt.hist(diam_um, bins=20, edgecolor='black', color='skyblue')
    plt.xlabel("Equivalent Pore Diameter (µm)")
    plt.ylabel("Count")
    plt.title("Pore Size Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../results/pore_size_histogram.png", dpi=300)
    plt.show()
    print("→ Saved histogram: ../results/pore_size_histogram.png")

