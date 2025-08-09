import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from skimage import io, filters, morphology, measure, img_as_bool

# === CONFIGURATION ===
IMAGE_PATH = "../images/sem_image.tif"
MASK_PATH = "../images/Mask.tif"
SCALE_UM_PER_PX = 30 / 145.377  # manually calibrated: 30 µm = 145.377 px
THRESH = 0.235  # chosen threshold from visual tuning
MIN_PORE_SIZE = 30  # pixel count filter for small objects
EXPORT_DIR = "../results"

# === LOAD & SEGMENT ===
image = io.imread(IMAGE_PATH, as_gray=True)
mask = img_as_bool(io.imread(MASK_PATH))
image = filters.gaussian(image, sigma=1)

# Threshold + mask
binary = (image < THRESH) & mask
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

# === COLOR BY DIAMETER ===
norm = plt.Normalize(vmin=min(diam_um), vmax=max(diam_um))
cmap = cm.spring  # bright + contrasty colormap
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

# Right: sorted histogram
sorted_idx = np.argsort(diam_um)
sorted_diam = diam_um[sorted_idx]
sorted_colors = colors[sorted_idx]

bars = ax2.bar(range(num_pores), sorted_diam, color=sorted_colors, edgecolor='black')
ax2.set_title("Sorted Pore Size Distribution", fontsize=16)
ax2.set_xlabel("Pore Index (sorted)", fontsize=14)
ax2.set_ylabel("Equivalent Diameter (µm)", fontsize=14)
ax2.tick_params(labelsize=12)
ax2.grid(True)

# Colorbar for visual reference
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label("Pore Diameter (µm)", fontsize=12)

# === EXPORT ===
plt.tight_layout()
fig_path = f"{EXPORT_DIR}/overlay_histogram_colormapped.png"
plt.savefig(fig_path, dpi=300)
plt.show()

# Save data to CSV
csv_path = f"{EXPORT_DIR}/pore_stats_colormapped.csv"
df = pd.DataFrame({
    "Pore Area (µm²)": areas_um2,
    "Equivalent Diameter (µm)": diam_um
})
df.to_csv(csv_path, index=False)
print(f"→ Saved: {fig_path}")
print(f"→ Saved: {csv_path}")

