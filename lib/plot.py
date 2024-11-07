# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# FUNCTION DEFINITION
# -----------------------------------------------------------------------------


def draw_arrows(file_name, mainImage, ref_center, shifts):
    fig, ax = plt.subplots(
        figsize=(10, 10 * mainImage.shape[0] / mainImage.shape[1])
    )  # Adjust figure size as needed

    # Extract coordinates and shifts
    x_ref = ref_center[..., 0]
    y_ref = ref_center[..., 1]
    x_shift = 12.5 * 10 * shifts[..., 0]
    y_shift = 12.5 * 10 * shifts[..., 1]

    # Plot arrows using plt.quiver
    # scale=1 and scale_units='xy' ensure the arrows represent shifts accurately
    ax.imshow(mainImage, cmap="gray", origin="upper")
    ax.quiver(
        x_ref,
        y_ref,
        x_shift,
        y_shift,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.003,
    )

    # Set plot limits to match the image size, or adjust as needed
    ax.set_xlim(0, mainImage.shape[1])
    ax.set_ylim(
        mainImage.shape[0], 0
    )  # Invert y-axis to match image coordinates

    # Show or save the plot
    # plt.axis('off')  # Remove axis for clean visualization
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close(fig)
