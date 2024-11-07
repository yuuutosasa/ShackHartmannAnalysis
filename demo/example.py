# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import cv2
from lib.plot import draw_arrows
from lib.shifts import (
    generate_ref_center,
    compute_shifts_moments,
    compute_shifts_NCC,
    compute_shifts_PC,
    pad_result,
)

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

# Read example image
img = cv2.imread("mainImage.png", 0)
img = (img.astype(np.float32) / 255.0) ** 3 * 255.0

ref_center = generate_ref_center(
    79.421, 1737.5, 27.45, 1226.4, 25.51, shape=(48, 66, 2)
)

print("Computing Moments...", end=" ", flush=True)
shifts_moments = compute_shifts_moments(img, ref_center)
print("Complete", flush=True)

print("Computing Phase Correlation...", end=" ", flush=True)
shifts_PC = compute_shifts_PC(img, ref_center)
print("Complete", flush=True)

print("Computing Normalized Cross Correlation...", end=" ", flush=True)
shifts_NCC = compute_shifts_NCC(img, ref_center)
print("Complete", flush=True)

print("Drawing arrows of moments result...", end=" ", flush=True)
draw_arrows("moments_arrows.png", img, ref_center, shifts_moments)
print("Complete", flush=True)

print("Drawing arrows of PC result...", end=" ", flush=True)
draw_arrows("PC_arrows.png", img, ref_center, shifts_PC)
print("Complete", flush=True)

print("Drawing arrows of NCC result...", end=" ", flush=True)
draw_arrows("NCC_arrows.png", img, ref_center, shifts_NCC)
print("Complete", flush=True)

shifts_moments = pad_result(shifts_PC, ref_center)
shifts_PC = pad_result(shifts_PC, ref_center)
shifts_NCC = pad_result(shifts_NCC, ref_center)
