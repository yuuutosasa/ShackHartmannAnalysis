# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from hswfs.plotting import disable_ticks
from hswfs.sensor import HSWFS
from hswfs.utils import get_unit_disk_meshgrid
from hswfs.zernike import Wavefront, eval_cartesian
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

# Define the output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define file paths for each method
methods = {
    "Moments": {
        "compute_func": compute_shifts_moments,
        "file_prefix": "moments",
    },
    "Phase_Correlation": {
        "compute_func": compute_shifts_PC,
        "file_prefix": "PC",
    },
    "NCC": {"compute_func": compute_shifts_NCC, "file_prefix": "NCC"},
}

# Read the example image
img = cv2.imread("mainImage.png", 0)
img = (img.astype(np.float32) / 255.0) ** 3 * 255.0

# Generate reference center
ref_center = generate_ref_center(79.421, 1737.5, 27.45, 1226.4, 25.51)

# Loop through each method
for method_name, method_data in methods.items():
    print(f"Computing {method_name}...", end=" ", flush=True)

    # Compute shifts based on method
    shifts = method_data["compute_func"](img, ref_center)
    print("Complete", flush=True)

    # Draw arrows and save the image for each method
    arrow_file = os.path.join(
        output_dir, f"{method_data['file_prefix']}_arrows.png"
    )
    print(f"Drawing arrows for {method_name}...", end=" ", flush=True)
    draw_arrows(arrow_file, img, ref_center, shifts)
    print("Complete", flush=True)

    # Pad result and save as numpy file if needed
    shifts_padded = pad_result(shifts, ref_center)
    np.save(
        os.path.join(output_dir, f"{method_data['file_prefix']}_array.npy"),
        shifts_padded,
    )

    # Set up wavefront sensor
    print(
        f"Setting up wavefront sensor for {method_name}...",
        end=" ",
        flush=True,
    )
    sensor = HSWFS(relative_shifts=shifts_padded)
    print("Done!", flush=True)

    # Fit the wavefront
    print(f"Fitting wavefront for {method_name}...", end=" ", flush=True)
    coefficients = sensor.fit_wavefront(n_zernike=9)
    wavefront = Wavefront(coefficients=coefficients)
    print("Done!", flush=True)

    # Plotting results
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[2, 1]
    )  # Two columns: wide left and narrow right
    ax_sensor = plt.subplot(gs[0, 0])  # Left column for array image

    # Right column: GridSpec with two rows for wavefront and coefficients
    right_grid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 1], height_ratios=[2, 1]
    )
    ax_wavefront = plt.subplot(right_grid[0, 0])
    ax_coeff = plt.subplot(right_grid[1, 0])

    # Plot shifts on wavefront sensor
    arrows = cv2.imread(arrow_file)
    ax_sensor.imshow(arrows)
    ax_sensor.set_title(f"Shifts on {method_name}")
    ax_sensor.set_aspect("equal")

    # Evaluate and plot the wavefront
    x_0, y_0 = get_unit_disk_meshgrid(resolution=512)
    wf_grid = eval_cartesian(wavefront.cartesian, x_0=x_0, y_0=y_0)
    limit = 1.1 * np.nanmax(np.abs(wf_grid))
    ax_wavefront.imshow(
        wf_grid,
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax_wavefront.set_title("Wavefront")

    # Plot the coefficients
    i_coeff = np.arange(len(coefficients))
    ax_coeff.bar(i_coeff, coefficients)
    ax_coeff.set_title("Coefficients")
    ax_coeff.set_xticks(i_coeff)

    # Disable ticks on all subplots
    disable_ticks(ax_sensor)
    disable_ticks(ax_wavefront)

    # Save each result plot
    result_file = os.path.join(
        output_dir, f"{method_data['file_prefix']}_result.png"
    )
    plt.tight_layout()
    plt.savefig(result_file, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    print(f"Done with {method_name}!", flush=True)
