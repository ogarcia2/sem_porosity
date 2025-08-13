import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from skimage import io, filters, morphology, measure

# === CONFIGURATION ===
IMAGE_PATH = "../images/Picture1cropped.png"
SCALE_UM_PER_PX = 30 / 145.377  # manually calibrated: 30 µm = 145.377 px
THRESH = 0.175  # chosen threshold from visual tuning
MIN_PORE_SIZE = 1  # pixel count filter for small objects "to remove noise"
EXPORT_DIR = "../results"

# === LOAD & SEGMENT ===
image = io.imread(IMAGE_PATH, as_gray=True)
image = filters.gaussian(image, sigma=0.5)

# Threshold without mask
binary = image < THRESH
binary = morphology.remove_small_objects(binary, min_size=MIN_PORE_SIZE)

# Label connected components
labeled = measure.label(binary)
props = measure.regionprops(labeled)

# === COMPUTE PORE METRICS ===
diam_um = np.array([p.equivalent_diameter * SCALE_UM_PER_PX for p in props])
areas_um2 = np.array([p.area * (SCALE_UM_PER_PX ** 2) for p in props])
num_pores = len(diam_um)
mean_diam = np.mean(diam_um)
median_diam = np.median(diam_um)
mean_area = np.mean(areas_um2)
total_area = np.sum(areas_um2)
image_area_um2 = image.shape[0] * image.shape[1] * (SCALE_UM_PER_PX ** 2)
porosity = (total_area / image_area_um2) * 100

# === PRINT SUMMARY ===
print("=== Pore Size Distribution Summary ===")
print(f"Total Pores:           {num_pores}")
print(f"Mean Diameter (µm):    {mean_diam:.2f}")
print(f"Median Diameter (µm):  {median_diam:.2f}")
print(f"Mean Area (µm²):       {mean_area:.2f}")
print(f"Porosity:              {porosity:.2f}%")
print("=====================================")

# === COLOR BY SIZE ===
norm = plt.Normalize(vmin=min(diam_um), vmax=max(diam_um))
cmap = cm.spring  # high-contrast color map
colors = cmap(norm(diam_um))

# === GENERATE COLORED OVERLAY ===
overlay_rgb = np.zeros((*labeled.shape, 3))
for i, color_rgb in enumerate(colors):
    overlay_rgb[labeled == (i + 1)] = color_rgb[:3]

alpha = 0.5
overlay = (1 - alpha) * np.stack([image]*3, axis=-1) + alpha * overlay_rgb

# === CREATE FIGURE ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plt.subplots_adjust(wspace=0.3)

# Left: overlay image
ax1.imshow(overlay)
ax1.set_title("Pore Segmentation (Colored by Diameter)", fontsize=16)
ax1.axis("off")

# Add 10 µm scale bar
bar_len_um = 10
bar_len_px = bar_len_um / SCALE_UM_PER_PX
bar_height = 8
bar_x = image.shape[1] - bar_len_px - 30
bar_y = image.shape[0] - 30
ax1.add_patch(Rectangle((bar_x, bar_y), bar_len_px, bar_height, color='white'))
ax1.text(bar_x + bar_len_px / 2, bar_y - 10, f"{bar_len_um} µm", color='white',
         ha='center', va='bottom', fontsize=12)

# Right: histogram by pore size bins
bin_edges = np.histogram_bin_edges(diam_um, bins=20)
counts, _ = np.histogram(diam_um, bins=bin_edges)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
color_bins = cmap(norm(bin_centers))
ax2.bar(bin_centers, counts, width=np.diff(bin_edges), align='center', color=color_bins, edgecolor='black')
ax2.set_title("Pore Size Histogram", fontsize=16)
ax2.set_xlabel("Equivalent Diameter (µm)", fontsize=14)
ax2.set_ylabel("Pore Count", fontsize=14)
ax2.tick_params(labelsize=12)
ax2.grid(True)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Pore Diameter (µm)", fontsize=12)

# === EXPORT ===
plt.tight_layout()
fig_path = f"{EXPORT_DIR}/overlay_histogram_colormapped.png"
plt.savefig(fig_path, dpi=300)
plt.show()

# Save CSV
csv_path = f"{EXPORT_DIR}/pore_stats_colormapped.csv"
df = pd.DataFrame({
    "Pore Area (µm²)": areas_um2,
    "Equivalent Diameter (µm)": diam_um
})
df.to_csv(csv_path, index=False)
print(f"→ Saved: {fig_path}")
print(f"→ Saved: {csv_path}")

