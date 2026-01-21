"""Check TIFF data statistics and metadata"""

import sys
from pathlib import Path
from bfio import BioReader
import numpy as np
import json

# Get input file from command line or use default
input_file = sys.argv[1] if len(sys.argv) > 1 else "files/sparc.tif"

br = BioReader(input_file, backend="bioformats")
print(f"=== TIFF File: {input_file} ===\n")

print("=== Basic Information ===")
print(f"Shape (Y, X, Z, C): {br.shape}")
print(f"Dtype: {br.dtype}")

print("\n=== Metadata ===")
# Print all metadata
if hasattr(br, "metadata") and br.metadata:
    print(json.dumps(br.metadata, indent=2, default=str))
else:
    print("No metadata available")

# Try to get physical dimensions
print("\n=== Physical Dimensions ===")
try:
    from xml.etree import ElementTree as ET

    if hasattr(br, "metadata") and br.metadata:
        # Try to extract physical sizes
        for key in br.metadata.keys():
            if "PhysicalSize" in str(key) or "physical" in str(key).lower():
                print(f"{key}: {br.metadata[key]}")
except Exception as e:
    print(f"Could not extract physical dimensions: {e}")

print("\n=== Data Statistics (sample) ===")
# Read a small sample to check values
sample = br.read(
    X=(0, min(100, br.shape[1])),
    Y=(0, min(100, br.shape[0])),
    Z=(0, min(10, br.shape[2])),
)
print(f"Sample shape: {sample.shape}")
print(f"Min: {sample.min()}")
print(f"Max: {sample.max()}")
print(f"Mean: {sample.mean():.2f}")
print(f"Std: {sample.std():.2f}")
print(f"Non-zero pixels: {np.count_nonzero(sample)}/{sample.size}")

print(f"\n=== Percentiles ===")
print(f"1%: {np.percentile(sample, 1):.2f}")
print(f"5%: {np.percentile(sample, 5):.2f}")
print(f"50%: {np.percentile(sample, 50):.2f}")
print(f"95%: {np.percentile(sample, 95):.2f}")
print(f"99%: {np.percentile(sample, 99):.2f}")

print(f"\n=== Per-Channel Statistics ===")
for c in range(sample.shape[-1]):
    ch_data = sample[..., c]
    print(
        f"Channel {c}: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean():.2f}, "
        f"non-zero={np.count_nonzero(ch_data)}/{ch_data.size}"
    )
