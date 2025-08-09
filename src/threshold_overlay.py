import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io, filters, morphology, measure, img_as_bool

# === CONFIG ===
IMAGE_PATH = "../images/sem_image.tif"
MASK_PATH = "../images/Mask.tif"
GAUSSIAN_SIGMA = 1
MIN_OBJECT_SIZE = 30
INITIAL_THRESH = 0.5
HFW_UM = 138.0  # full width in microns

# === LOAD DATA ===
image = io.imread(IMAGE_PATH, as_gray=True)
image = filters.gaussian(image, sigma=GAUSSIAN_SIGMA)
mask = img_as_bool(io.imread(MASK_PATH))
image_masked = np.where(mask, image, np.nan)

# Calibrate pixel size
height, width = image.shape
scale_um_per_px = HFW_UM / width

# === INITIAL SEGMENTATION ===
def segment(thresh):
    binary = image_masked < thresh
    binary = np.where(mask, binary, False)
    binary = morphology.remove_small_objects(binary, min_size=MIN_OBJECT_SIZE)
    return binary

seg = segment(INITIAL_THRESH)

# === FIGURE SETUP ===
fig, (ax_img, ax_stats) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [4, 1]})
plt.subplots_adjust(bottom=0.2)

# Image display
ax_img.imshow(image, cmap='gray')
overlay_data = np.zeros((*seg.shape, 4))
overlay_data[seg] = [0, 1, 1, 0.4]
overlay = ax_img.imshow(overlay_data)
ax_img.set_title(f"Cyan Overlay (Threshold = {INITIAL_THRESH:.2f})")
ax_img.axis("off")

# Stats display
text_box = ax_stats.text(0.05, 0.95, "", va='top', ha='left', fontsize=12)
ax_stats.axis("off")

def update_stats_display(segmentation, thresh):
    labeled = measure.label(segmentation)
    props = measure.regionprops(labeled)

    areas_um2 = [p.area * (scale_um_per_px ** 2) for p in props]
    porosity = (np.sum(segmentation) / np.sum(mask)) * 100

    summary_text = (
        f"Threshold: {thresh:.3f}\n"
        f"Pore count: {len(areas_um2)}\n"
        f"Porosity: {porosity:.2f}%\n"
        f"Mean area: {np.mean(areas_um2):.2f} µm²"
    )
    return summary_text

text_box.set_text(update_stats_display(seg, INITIAL_THRESH))

# === SLIDER ===
slider_ax = plt.axes([0.25, 0.05, 0.5, 0.03])
slider = Slider(slider_ax, "Threshold", 0.0, 1.0, valinit=INITIAL_THRESH, valstep=0.01)

# === CALLBACK ===
def update(thresh):
    seg_new = segment(thresh)

    rgba = np.zeros((*seg_new.shape, 4))
    rgba[seg_new] = [0, 1, 1, 0.4]
    overlay.set_data(rgba)

    ax_img.set_title(f"Cyan Overlay (Threshold = {thresh:.2f})")
    text_box.set_text(update_stats_display(seg_new, thresh))
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()

