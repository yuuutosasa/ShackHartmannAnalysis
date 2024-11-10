# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import cv2
from skimage.feature import match_template

# -----------------------------------------------------------------------------
# FUNCTION DEFINITION
# -----------------------------------------------------------------------------


def generate_ref_center(x_min, x_max, y_min, y_max, lens_pitch):
    x_ref_repeat = np.arange(x_min, x_max + 1, lens_pitch)
    y_ref_repeat = np.arange(y_min, y_max + 1, lens_pitch)

    ref_center = np.ndarray((len(x_ref_repeat) * len(y_ref_repeat), 2))

    x_grid, y_grid = np.meshgrid(x_ref_repeat, y_ref_repeat)

    # Combine the grids into a list of coordinate pairs
    shape = (len(y_ref_repeat), len(x_ref_repeat), 2)
    ref_center = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    ref_center = ref_center.reshape(shape)
    return ref_center


def compute_shifts_moments(img, ref_center, grid=25):
    H, W, _ = ref_center.shape
    th = grid / 2

    shifts = np.full((H, W, 2), np.NaN)

    for i in range(H):
        for j in range(W):
            img_tar = cv2.getRectSubPix(
                img, (grid, grid), (ref_center[i, j, 0], ref_center[i, j, 1])
            )
            mm = cv2.moments(img_tar)
            if mm["m00"] > 100:
                shifts[i, j, 0] = mm["m10"] / mm["m00"] - th
                shifts[i, j, 1] = mm["m01"] / mm["m00"] - th
    shifts /= th
    return shifts


def compute_shifts_PC(img, ref_center, grid=25):
    H, W, _ = ref_center.shape
    th = grid / 2

    corr_shift = np.full((H, W, 2), np.NaN)

    # Prepare Hanning window
    hann_window = (
        np.hanning(grid)[:, np.newaxis] * np.hanning(grid)[np.newaxis, :]
    )
    hann_window /= np.max(hann_window)
    hann_window = np.pad(
        hann_window,
        ((grid // 2, grid // 2), (grid // 2, grid // 2)),
        mode="constant",
    )

    img_ref = cv2.getRectSubPix(
        img,
        (grid, grid),
        (ref_center[H // 2, W // 2, 0], ref_center[H // 2, W // 2, 1]),
    )
    img_ref = np.pad(
        img_ref,
        ((grid // 2, grid // 2), (grid // 2, grid // 2)),
        mode="constant",
    )
    for i in range(H):
        for j in range(W):
            img_tar = cv2.getRectSubPix(
                img, (grid, grid), (ref_center[i, j, 0], ref_center[i, j, 1])
            )
            mm = cv2.moments(img_tar)
            if mm["m00"] > 100:
                # Pad the target image
                img_tar = np.pad(
                    img_tar,
                    ((grid // 2, grid // 2), (grid // 2, grid // 2)),
                    mode="constant",
                )
                displacement, _ = cv2.phaseCorrelate(
                    img_ref.astype(np.float64),
                    img_tar.astype(np.float64),
                    window=hann_window.astype(np.float64),
                )
                if np.max(np.linalg.norm(displacement)) <= th:
                    corr_shift[i, j, 0] = displacement[0]
                    corr_shift[i, j, 1] = displacement[1]
    corr_shift /= th
    return corr_shift


def compute_shifts_NCC(img, ref_center, grid=25):
    H, W, _ = ref_center.shape
    th = grid / 2

    ncc_shift = np.full((H, W, 2), np.NaN)

    img_ref = cv2.getRectSubPix(
        img,
        (grid, grid),
        (ref_center[H // 2, W // 2, 0], ref_center[H // 2, W // 2, 1]),
    )
    img_ref = np.pad(
        img_ref,
        ((grid // 2, grid // 2), (grid // 2, grid // 2)),
        mode="constant",
    )
    for i in range(H):
        for j in range(W):
            img_tar = cv2.getRectSubPix(
                img, (grid, grid), (ref_center[i, j, 0], ref_center[i, j, 1])
            )
            mm = cv2.moments(img_tar)
            if mm["m00"] > 100:
                # Pad the target image
                img_tar = np.pad(
                    img_tar,
                    ((25 // 2, 25 // 2), (25 // 2, 25 // 2)),
                    mode="constant",
                )
                displacement = subpixel_ncc(
                    img_tar.astype(np.float64), img_ref.astype(np.float64)
                )
                if np.max(np.linalg.norm(displacement)) <= 12.75:
                    ncc_shift[i, j, 0] = displacement[0]
                    ncc_shift[i, j, 1] = displacement[1]
    ncc_shift /= th
    return ncc_shift


def subpixel_ncc(image1, image2):
    """
    Calculate subpixel displacement between two images using enhanced NCC techniques.

    Parameters:
    image1 (ndarray): The first image (reference).
    image2 (ndarray): The second image (shifted).
    sigma (float): Standard deviation for Gaussian smoothing to reduce noise.

    Returns:
    tuple: Displacement in (x, y) with subpixel accuracy.
    """
    # Compute normalized cross-correlation with padding
    result = match_template(image1, image2, pad_input=True)

    # Pad the result to handle edge cases when finding the peak
    padded_result = np.pad(result, ((1, 1), (1, 1)), mode="constant")

    # Get the integer location of the maximum correlation
    ij = np.unravel_index(np.argmax(padded_result), padded_result.shape)
    y, x = ij

    # Verify if a 3x3 neighborhood can be extracted
    if (
        y - 1 < 0
        or y + 1 > padded_result.shape[0]
        or x - 1 < 0
        or x + 1 > padded_result.shape[1]
    ):
        # Return integer displacement if neighborhood extraction fails
        center = (padded_result.shape[1] // 2, padded_result.shape[0] // 2)
        return (x - center[0], y - center[1])

    # Extract a 3x3 neighborhood around the peak for subpixel interpolation
    neighborhood = padded_result[y - 1 : y + 2, x - 1 : x + 2]

    # Compute subpixel peak using parabolic interpolation
    dx = (
        -0.5
        * (neighborhood[1, 2] - neighborhood[1, 0])
        / (neighborhood[1, 2] - 2 * neighborhood[1, 1] + neighborhood[1, 0])
    )
    dy = (
        -0.5
        * (neighborhood[2, 1] - neighborhood[0, 1])
        / (neighborhood[2, 1] - 2 * neighborhood[1, 1] + neighborhood[0, 1])
    )

    # Adjust subpixel peak position to account for the padding offset
    subpixel_x = x + dx
    subpixel_y = y + dy

    # Calculate displacement from the image center
    center = (padded_result.shape[1] // 2, padded_result.shape[0] // 2)
    displacement = (subpixel_x - center[0], subpixel_y - center[1])

    return displacement


def pad_result(shifts, ref_center):
    H, W, _ = ref_center.shape
    padded_shifts = np.full((W, W, 2), np.NaN)
    pad_top = (W - H) // 2 + 1
    padded_shifts[pad_top : pad_top + H, ...] = shifts
    return padded_shifts
