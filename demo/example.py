# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import cv2
from matplotlib import pyplot as plt
from lib.grid import draw_grid
from lib.shifts import cut2zero, shifts

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

# Read example image
img_shifts = cv2.imread("example_image.jpg")

# Output example image with subaperture grids with the center
img_grid = draw_grid(img_shifts)
plt.imshow(img_grid)
plt.savefig("grid.png")

# Prepare a grayscale image
img_gray = img_shifts[..., 0]
img_gray = cut2zero(img_gray)

# Compute shifts
shifts = shifts(img_gray)
np.save("shifts_array.npy", shifts)
print(shifts[0, :])
