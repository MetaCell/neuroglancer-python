"""
Quick TIFF visualization to diagnose "black image" issues
"""

from pathlib import Path
from bfio import BioReader
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = Path("files/sparc.tif")

print(f"Opening: {INPUT_PATH}")
br = BioReader(str(INPUT_PATH), backend="bioformats")

print(f"\n=== Image Info ===")
print(f"Shape: {br.shape} (Y, X, Z, C)")
print(f"Dtype: {br.dtype}")

# Read middle slice
z_mid = br.shape[2] // 2
print(f"\nReading middle Z slice ({z_mid})...")
data = br.read(Z=(z_mid, z_mid + 1))
data = data[:, :, 0, :]  # Remove Z dimension

print(f"\n=== Statistics ===")
print(f"Overall: min={data.min()}, max={data.max()}, mean={data.mean():.2f}")
print(
    f"Non-zero pixels: {np.count_nonzero(data)}/{data.size} ({100*np.count_nonzero(data)/data.size:.1f}%)"
)

# Per-channel stats
num_channels = data.shape[-1]
print(f"\nPer-channel:")
for c in range(num_channels):
    ch = data[:, :, c]
    nz = np.count_nonzero(ch)
    print(f"  Channel {c}:")
    print(f"    Range: [{ch.min()}, {ch.max()}]")
    print(f"    Mean: {ch.mean():.2f}, Std: {ch.std():.2f}")
    print(f"    Non-zero: {nz}/{ch.size} ({100*nz/ch.size:.1f}%)")
    if nz > 0:
        nonzero_vals = ch[ch > 0]
        print(
            f"    Non-zero values: min={nonzero_vals.min()}, max={nonzero_vals.max()}, mean={nonzero_vals.mean():.2f}"
        )

# Visualize
fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))
if num_channels == 1:
    axes = [axes]

for c in range(num_channels):
    ch_data = data[:, :, c]

    # Try different contrast settings
    vmin, vmax = (
        np.percentile(ch_data[ch_data > 0], [1, 99]) if np.any(ch_data > 0) else (0, 1)
    )

    im = axes[c].imshow(ch_data, cmap="gray", vmin=vmin, vmax=vmax)
    axes[c].set_title(
        f"Channel {c}\nRange: [{ch_data.min()}, {ch_data.max()}]\nDisplay: [{vmin:.1f}, {vmax:.1f}]"
    )
    axes[c].axis("off")
    plt.colorbar(im, ax=axes[c])

plt.suptitle(
    f"{INPUT_PATH.name} - Middle Z slice\n(contrast: 1st-99th percentile of non-zero values)",
    fontsize=12,
)
plt.tight_layout()
plt.savefig("tiff_preview.png", dpi=150, bbox_inches="tight")
print(f"\nSaved preview to: tiff_preview.png")
print("\nTo view in napari with proper contrast:")
print(f"  1. Open napari")
print(f"  2. Load the file")
print(f"  3. Adjust contrast limits to approximately [{vmin:.1f}, {vmax:.1f}]")
plt.show()
