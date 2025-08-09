# ~/Documents/LLZO_Diffusivity/sem_porosity/threshold_tuner.py

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io, filters, morphology
import numpy as np
import os

# === CONFIG ===
IMAGE_PATH = "../images/sem_image.tif"
GAUSSIAN_SIGMA = 1
MIN_OBJECT_SIZE = 30  # Minimum pixel size to keep

# === LOAD IMAGE ===
image = io.imread(IMAGE_PATH, as_gray=True)
image = filters.gaussian(image, sigma=GAUSSIAN_SIGMA)

# === INITIAL THRESHOLD ===
initial_thresh = 0.5
binary = image < initial_thresh
binary = morphology.remove_small_objects(binary, min_size=MIN_OBJECT_SIZE)

# === PLOT SETUP ===
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.2)

# Show original
ax0 = axes[0]
ax0.imshow(image, cmap="gray")
ax0.set_title("Original SEM")
ax0.axis("off")

# Show binary mask
ax1 = axes[1]
mask_im = ax1.imshow(binary, cmap="gray")
ax1.set_title(f"Pores (Threshold={initial_thresh:.2f})")
ax1.axis("off")

# === SLIDER ===
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
slider = Slider(ax_slider, "Threshold", 0.0, 1.0, valinit=initial_thresh, valstep=0.01)

def update(val):
    thresh = slider.val
    bin_mask = image < thresh
    bin_mask = morphology.remove_small_objects(bin_mask, min_size=MIN_OBJECT_SIZE)
    mask_im.set_data(bin_mask)
    ax1.set_title(f"Pores (Threshold={thresh:.2f})")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()

