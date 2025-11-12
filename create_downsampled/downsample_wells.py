"""
Downsample N5 Dataset to OME-Zarr Pyramids (Well-by-Well)

This script reads a dataset in N5 format and outputs downsampled pyramids
in OME-Zarr format, processing one well at a time.

Each well is 540 x 540 pixels. The script validates wells, assigns IDs by
row and column, and creates downsampled pyramids at configurable factors.
"""

# %% import and definition
import itertools as itt
import logging
import shutil
from pathlib import Path

import dask.array as da
import numpy as np
import tensorstore as ts
import zarr
from ome_zarr.dask_utils import resize
from ome_zarr.format import CurrentFormat
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata

INPUT_PATH = Path("data/cppx158_cpp2388/s0")  # Path to N5 dataset
OUTPUT_PATH = Path("data/output_zarr")  # Output directory for OME-Zarr pyramids
WELL_SIZE = (540, 540)  # Well dimensions
DOWNSAMPLE_FACTORS = [2, 4]  # Downsampling factors (1=full res, arbitrary integers)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Input path: {INPUT_PATH}")
logger.info(f"Output path: {OUTPUT_PATH}")
logger.info(f"Well size: {WELL_SIZE}")
logger.info(f"Downsample factors: {DOWNSAMPLE_FACTORS}")


def validate_well(shape: tuple, well_size: tuple) -> tuple:
    x_size, y_size = shape[0], shape[1]
    well_width, well_height = well_size
    if x_size % well_width != 0:
        raise ValueError(
            f"Dataset X dimension ({x_size}) is not divisible by "
            f"well width ({well_width}). Remainder: {x_size % well_width}"
        )
    if y_size % well_height != 0:
        raise ValueError(
            f"Dataset Y dimension ({y_size}) is not divisible by "
            f"well height ({well_height}). Remainder: {y_size % well_height}"
        )
    logger.info("✓ Dataset dimensions are divisible by well size")
    n_cols = x_size // well_width
    n_rows = y_size // well_height
    logger.info(f"  Number of wells: {n_cols} cols x {n_rows} rows")
    return n_rows, n_cols


def downsample_pyramid_on_disk(
    source_data, image_path, downsample_factors, axes_in, axes_out, fmt
):
    """
    Downsample pyramid levels on disk using resize directly from source data.
    Creates pyramid levels with arbitrary downsample factors directly from input.

    Args:
        source_data: Source numpy/dask array (full resolution data)
        image_path: Path to the image directory
        downsample_factors: List of downsample factors (e.g., [1, 2, 4, 8])
                          Factor of 1 means no downsampling (original resolution)
        axes_in: Input axis metadata for OME-NGFF
        axes_out: Output axis metadata for OME-NGFF
        fmt: CurrentFormat instance
    """
    # Convert to dask array if needed
    if not isinstance(source_data, da.Array):
        source_data = da.from_array(source_data)
    # Calculate transpose order by comparing axes_in and axes_out by name
    axes_in_names = [axis["name"] for axis in axes_in]
    axes_out_names = [axis["name"] for axis in axes_out]
    transpose_order = [axes_in_names.index(name) for name in axes_out_names]
    store = parse_url(str(image_path), mode="w", fmt=fmt).store
    root = zarr.group(store=store)
    metadata = []
    for level, factor in enumerate(downsample_factors):
        # Use resize directly from ome-zarr
        if factor == 1:
            output = source_data
        else:
            new_shape = get_downsample_shape(axes_in, source_data.shape, factor)
            output = resize(source_data, new_shape, preserve_range=True)
        # Transpose the output array to match axes_out order
        output = output.transpose(transpose_order)
        # Write to zarr
        options = {
            "chunk_key_encoding": fmt.chunk_key_encoding,
            "dimension_names": axes_out_names,
        }
        da.to_zarr(
            arr=output,
            url=image_path,
            component=str(level),
            zarr_format=fmt.zarr_format,
            **options,
        )
        # Create coordinate transformations and datasets metadata
        # Scale should be in the output axis order
        metadata.append(
            {
                "path": str(level),
                "coordinateTransformations": create_transform(axes_out, factor),
            }
        )
    # Write multiscales metadata
    write_multiscales_metadata(root, metadata, axes=axes_out)


def create_transform(axes, factor, spatial_names=["x", "y"]):
    scale = [1.0] * len(axes)
    for axis_name in ("x", "y"):
        if axis_name in axes:
            axis_idx = axes.index(axis_name)
            scale[axis_idx] = float(factor)
    return [
        {
            "type": "scale",
            "scale": scale,
        }
    ]


def get_downsample_shape(axes, shape, factor):
    spatial_axes = [i for i, axis in enumerate(axes) if axis["name"] in ("x", "y")]
    new_shape = list(shape)
    for axis_idx in spatial_axes:
        new_shape[axis_idx] = shape[axis_idx] // factor
    return tuple(new_shape)


# %% Load the input dataset
input_dims = ["X", "Y", "time", "z", "channel_id"]
input_data = ts.open(
    {
        "driver": "n5",
        "kvstore": {"driver": "file", "path": str(INPUT_PATH)},
    }
).result()
logger.info(f"Input data shape: {input_data.shape}")
logger.info(f"Input data dtype: {input_data.dtype}")

# %% Process each well
if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)
OUTPUT_PATH.mkdir(parents=True)
fmt = CurrentFormat()
axes_in = [
    {"name": "x", "type": "space", "unit": "pixel"},
    {"name": "y", "type": "space", "unit": "pixel"},
    {"name": "t", "type": "time"},
    {"name": "z", "type": "space", "unit": "pixel"},
    {"name": "c", "type": "channel"},
]
axes_out = [
    {"name": "t", "type": "time"},
    {"name": "c", "type": "channel"},
    {"name": "x", "type": "space", "unit": "pixel"},
    {"name": "y", "type": "space", "unit": "pixel"},
    {"name": "z", "type": "space", "unit": "pixel"},
]
n_rows, n_cols = validate_well(input_data.shape, WELL_SIZE)
logger.info(f"Processing {n_rows} rows x {n_cols} cols = {n_rows * n_cols} wells")
logger.info("Starting well-by-well processing using ome-zarr...")
for row, col in itt.product(range(n_rows), range(n_cols)):
    well_id = f"well_r{row:02d}_c{col:02d}"
    logger.info(f"Processing {well_id} (row={row}, col={col})")
    # Calculate well boundaries
    well_width, well_height = WELL_SIZE
    x_start = col * well_width
    x_end = x_start + well_width
    y_start = row * well_height
    y_end = y_start + well_height
    # Extract well data (numpy array slicing)
    well_data = input_data[x_start:x_end, y_start:y_end, :, :, :]
    # Create output path
    well_output_path = OUTPUT_PATH / well_id
    well_output_path.mkdir(parents=True, exist_ok=True)
    # Create pyramid levels directly from source data
    downsample_pyramid_on_disk(
        np.array(well_data),
        str(well_output_path),
        DOWNSAMPLE_FACTORS,
        axes_in,
        axes_out,
        fmt,
    )
    logger.debug(f"  ✓ {well_id} written successfully")
logger.info("✓ All wells processed successfully!")
