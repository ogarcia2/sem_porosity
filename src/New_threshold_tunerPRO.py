import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from skimage import io, filters, morphology, measure
import numpy as np

# === CONFIG ===
IMAGE_PATH = "../images/Picture1cropped.png"
GAUSSIAN_SIGMA_INIT = 0.5
MIN_OBJECT_SIZE = 1
SCALE_UM_PER_PX = 30 / 145.377

# === LOAD IMAGE ===
image_orig = io.imread(IMAGE_PATH, as_gray=True)
IMAGE_HEIGHT, IMAGE_WIDTH = image_orig.shape

# === PROCESS ===
def process(thresh, sigma, closing, dilation, fill):
    image = filters.gaussian(image_orig, sigma=sigma)
    binary = image < thresh

    if closing:
        binary = morphology.closing(binary, morphology.disk(1))
    if dilation:
        binary = morphology.dilation(binary, morphology.disk(1))
    if fill:
        binary = morphology.remove_small_holes(binary, area_threshold=64)

    binary = morphology.remove_small_objects(binary, min_size=MIN_OBJECT_SIZE)
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)

    total_pore_area = np.sum([p.area for p in props]) * (SCALE_UM_PER_PX ** 2)
    image_area = IMAGE_WIDTH * IMAGE_HEIGHT * (SCALE_UM_PER_PX ** 2)
    porosity = (total_pore_area / image_area) * 100

    return binary, len(props), porosity

# === FIGURE SETUP ===
fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [4, 1]})
plt.subplots_adjust(bottom=0.25)

# Initial process
binary, count, porosity = process(0.5, GAUSSIAN_SIGMA_INIT, False, False, False)

img_display = ax_img.imshow(binary, cmap="gray")
ax_img.set_title("Pore Segmentation", fontsize=14)
ax_img.axis("off")

text_display = ax_txt.text(0.01, 0.95, "", fontsize=12, va='top', ha='left')
ax_txt.axis("off")

# === SLIDERS ===
thresh_ax = plt.axes([0.25, 0.15, 0.5, 0.03])
sigma_ax = plt.axes([0.25, 0.10, 0.5, 0.03])

thresh_slider = Slider(thresh_ax, "Threshold", 0.0, 1.0, valinit=0.5, valstep=0.005)
sigma_slider = Slider(sigma_ax, "Gaussian σ", 0.0, 3.0, valinit=GAUSSIAN_SIGMA_INIT, valstep=0.1)

# === CHECKBOXES ===
check_ax = plt.axes([0.05, 0.5, 0.15, 0.2])
check = CheckButtons(check_ax, ["Closing", "Dilation", "Fill Holes"], [False, False, False])

# === UPDATE FUNCTION ===
def update(val):
    thresh = thresh_slider.val
    sigma = sigma_slider.val
    closing, dilation, fill = check.get_status()
    binary, count, porosity = process(thresh, sigma, closing, dilation, fill)
    img_display.set_data(binary)
    ax_img.set_title(f"Threshold = {thresh:.3f}", fontsize=14)
    text_display.set_text(f"Pores: {count}\nPorosity: {porosity:.2f}%")
    fig.canvas.draw_idle()

# Bind updates
thresh_slider.on_changed(update)
sigma_slider.on_changed(update)
check.on_clicked(update)

update(None)
plt.show()

