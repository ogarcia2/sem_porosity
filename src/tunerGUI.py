import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from skimage import io, filters, morphology, img_as_bool

# === CONFIG ===
IMAGE_PATH = "../images/sem_image.tif"
MASK_PATH = "../images/Mask.tif"
MIN_OBJECT_SIZE = 30
SCALE_UM_PER_PX = 30 / 145.377

# === LOAD ===
image = io.imread(IMAGE_PATH, as_gray=True)
mask = img_as_bool(io.imread(MASK_PATH))

# === INITIAL PARAMETERS ===
def segment(thresh, sigma, closing, dilate, fill):
    img_blur = filters.gaussian(image, sigma=sigma)
    binary = (img_blur < thresh) & mask

    if closing:
        binary = morphology.closing(binary, morphology.disk(1))
    if dilate:
        binary = morphology.dilation(binary, morphology.disk(1))
    if fill:
        binary = morphology.remove_small_holes(binary, area_threshold=64)

    binary = morphology.remove_small_objects(binary, min_size=MIN_OBJECT_SIZE)
    return binary, img_blur

# === PLOT SETUP ===
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.3, bottom=0.25)

overlay_im = ax.imshow(image, cmap='gray')
seg_overlay = ax.imshow(np.zeros_like(image), cmap='spring', alpha=0.5)
ax.axis("off")
ax.set_title("Interactive Threshold Tuner")

# === SLIDERS ===
thresh_ax = plt.axes([0.3, 0.15, 0.55, 0.03])
thresh_slider = Slider(thresh_ax, "Threshold", 0.0, 1.0, valinit=0.25, valstep=0.005)

sigma_ax = plt.axes([0.3, 0.1, 0.55, 0.03])
sigma_slider = Slider(sigma_ax, "Gaussian σ", 0.0, 3.0, valinit=1.0, valstep=0.1)

# === TOGGLES ===
check_ax = plt.axes([0.05, 0.4, 0.2, 0.2])
check = CheckButtons(check_ax, ["Closing", "Dilation", "Fill Holes"], [False, False, False])

# === UPDATE ===
def update(val):
    thresh = thresh_slider.val
    sigma = sigma_slider.val
    closing, dilate, fill = check.get_status()
    binary, img_blur = segment(thresh, sigma, closing, dilate, fill)
    seg_overlay.set_data(binary)
    fig.canvas.draw_idle()

thresh_slider.on_changed(update)
sigma_slider.on_changed(update)
check.on_clicked(update)

update(0.25)
plt.show()

