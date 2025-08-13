# === PURPOSE ===
# Sweep through thresholds to find which value (between 0.09 and 0.16) gives ~11% porosity.
# Adjustable Gaussian blur and morphological parameters are set as variables up top.

import numpy as np
from skimage import io, filters, morphology, measure, img_as_bool

# === CONFIGURATION ===
IMAGE_PATH = "../images/sem_image.tif"
MASK_PATH = "../images/Mask.tif"
SCALE_UM_PER_PX = 30 / 145.377
MIN_OBJECT_SIZE = 30

# === TOGGLE OPTIONS ===
GAUSSIAN_SIGMA = .5         # Change smoothing here
APPLY_CLOSING = False        # Set True to enable morphological closing
APPLY_DILATION = False       # Set True to enable dilation
FILL_HOLES = False           # Set True to fill small internal gaps

# === THRESHOLD RANGE ===
thresholds = np.arange(0.09, 0.265, 0.005)  # 0.09 to 0.16 (inclusive)

# === LOAD ===
image = io.imread(IMAGE_PATH, as_gray=True)
mask = img_as_bool(io.imread(MASK_PATH))
image = filters.gaussian(image, sigma=GAUSSIAN_SIGMA)

# === IMAGE AREA ===
image_area_um2 = image.shape[0] * image.shape[1] * (SCALE_UM_PER_PX ** 2)

# === ANALYSIS LOOP ===
print("Threshold\tPorosity (%)\tNum Pores")
for thresh in thresholds:
    binary = (image < thresh) & mask

    if APPLY_CLOSING:
        binary = morphology.closing(binary, morphology.disk(1))
    if APPLY_DILATION:
        binary = morphology.dilation(binary, morphology.disk(1))
    if FILL_HOLES:
        binary = morphology.remove_small_holes(binary, area_threshold=64)

    binary = morphology.remove_small_objects(binary, min_size=MIN_OBJECT_SIZE)

    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    total_area = np.sum([p.area for p in props]) * (SCALE_UM_PER_PX ** 2)
    porosity = (total_area / image_area_um2) * 100

    print(f"{thresh:.3f}\t\t{porosity:.2f}\t\t{len(props)}")


