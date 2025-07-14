# %% Import packages

import itertools
import math
from pathlib import Path
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import touch, Vec
import zarr
from neuroglancer.downsample import downsample_with_averaging

# %% Define the path to the files

HERE = Path(__file__).parent

# Paths to change
INPUTFOLDER = Path("")
OUTPUT_PATH = Path("")

# Other settings
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
OVERWRITE = False
NUM_MIPS = 5
MIP_CUTOFF = 0  # To save time you can start at the lowest resolution and work up
NUM_CHANNELS = 4  # For less memory usage (can't be 1 right now though)
NUM_ROWS = 3
NUM_COLS = 6
ALLOW_NON_ALIGNED_WRITE = True

# %% Load the data
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# You don't need all files if you download them on demand
# See get_file_for_row_col function
all_files = list(INPUTFOLDER.glob("**/*.zarr"))


def get_file_for_row_col(row, col):
    """Get the file path for a specific row and column."""
    if row < 0 or row >= NUM_ROWS or col < 0 or col >= NUM_COLS:
        raise ValueError("Row and column indices must be within the defined grid.")
    index = row * NUM_COLS + col
    # You could also download the file here and then delete it after use
    return all_files[index] if index < len(all_files) else None


def load_zarr_store(file_path):
    zarr_store = zarr.open(file_path, mode="r")
    return zarr_store


def load_data_from_zarr_store(zarr_store):
    # Input is in Z, T, C, Y, X order
    data = zarr_store[
        :,  # T
        :,  # Z
        :NUM_CHANNELS,  # C
        :,  # Y
        :,  # X
    ]
    # The original timestamp was 65535, can be filterd out
    data = np.where(data == 65535, 0, data)
    data = np.squeeze(data)  # Remove any singleton dimensions
    print("Loaded data shape:", data.shape)
    # Then we permute to XYTCZ
    data = np.transpose(data, (-1, -2, 0, 1))  # Permute to XYTCZ
    return data


zarr_store = load_zarr_store(all_files[0])

# %% Inspect the data
shape = zarr_store.shape
# Input is in Z, T, C, Y, X order
# Want XYTCZ order
single_file_xyz_shape = [shape[4], shape[3], shape[1]]
size_x = 1
size_y = 1
size_z = 1
# Here, T and Z are kind of transferrable.
# The reason is because the z dimension in neuroglancer is the time dimension
# from the raw original data.
# So both terms might be used to represent the same thing.
# It's a bit unusual that t is being used as the z dimension,
# but otherwise you can't do volume rendering in neuroglancer.

num_channels = min(shape[2], NUM_CHANNELS)  # Limit to NUM_CHANNELS for memory usage
data_type = "uint16"
chunk_size = [64, 64, 32] # chunk size in neuroglancer
# The chunk size remains fixed across all mips, but at higher mips
# the data will be downsampled, so the effective chunk size will be larger.
# Because we use precomputed data format
# Every chunk has to have all channels included

volume_size = [
    single_file_xyz_shape[0] * NUM_ROWS,
    single_file_xyz_shape[1] * NUM_COLS,
    single_file_xyz_shape[2],
]  # XYZ (T)
print("Volume size:", volume_size)

# %% Setup the cloudvolume info
info = CloudVolume.create_new_info(
    num_channels=num_channels,
    layer_type="image",
    data_type=data_type,
    encoding="raw",
    resolution=[size_x, size_y, size_z],
    voxel_offset=[0, 0, 0],
    chunk_size=chunk_size,
    volume_size=volume_size,
    max_mip=NUM_MIPS - 1,
    factor=Vec(2, 2, 2),
)
vol = CloudVolume(
    "file://" + str(OUTPUT_PATH),
    info=info,
    mip=0,
    non_aligned_writes=ALLOW_NON_ALIGNED_WRITE,
    fill_missing=True,
)
vol.commit_info()
vol.provenance.description = "Example data conversion"
vol.commit_provenance()
del vol

# %% Create the volumes for each mip level and hold progress
vols = [
    CloudVolume(
        "file://" + str(OUTPUT_PATH),
        mip=i,
        compress=False,
        non_aligned_writes=ALLOW_NON_ALIGNED_WRITE,
        fill_missing=True,
    )
    for i in range(NUM_MIPS)
]
progress_dir = OUTPUT_PATH / "progress"
progress_dir.mkdir(exist_ok=True)

# %% Functions for moving data
shape = volume_size
chunk_shape = np.array([1500, 936, 687])  # this is for reading data
num_chunks_per_dim = np.ceil(shape / chunk_shape).astype(int)


def process(args):
    x_i, y_i, z_i = args
    file_to_load = get_file_for_row_col(x_i, y_i)
    print(f"Processing {file_to_load} at coordinates ({x_i}, {y_i}, {z_i})")
    loaded_zarr_store = load_zarr_store(file_to_load)
    start = [x_i * chunk_shape[0], y_i * chunk_shape[1], z_i * chunk_shape[2]]
    end = [
        min((x_i + 1) * chunk_shape[0], shape[0]),
        min((y_i + 1) * chunk_shape[1], shape[1]),
        min((z_i + 1) * chunk_shape[2], shape[2]),
    ]
    f_name = progress_dir / f"{start[0]}-{end[0]}_{start[1]}-{end[1]}.done"
    print(f"Processing chunk: {start} to {end}, file: {f_name}")
    if f_name.exists() and not OVERWRITE:
        return
    rawdata = load_data_from_zarr_store(loaded_zarr_store)
    for mip_level in reversed(range(MIP_CUTOFF, NUM_MIPS)):
        if mip_level == 0:
            downsampled = rawdata
            ds_start = start
            ds_end = end
        else:
            factor = 2**mip_level
            factor_tuple = (factor, factor, factor, 1)
            ds_start = [int(np.round(s / (2**mip_level))) for s in start]
            bounds_from_end = [int(math.ceil(e / (2**mip_level))) for e in end]
            downsample_shape = [
                int(math.ceil(s / f)) for s, f in zip(rawdata.shape, factor_tuple)
            ]
            ds_end_est = [s + d for s, d in zip(ds_start, downsample_shape)]
            ds_end = [max(e1, e2) for e1, e2 in zip(ds_end_est, bounds_from_end)]
            print("DS fill", ds_start, ds_end)
            downsampled = downsample_with_averaging(rawdata, factor_tuple)
            print("Downsampled shape:", downsampled.shape)

        vols[mip_level][
            ds_start[0] : ds_end[0], ds_start[1] : ds_end[1], ds_start[2] : ds_end[2]
        ] = downsampled
    touch(f_name)


# %% Try with a single chunk to see if it works
# x_i, y_i, z_i = 0, 0, 0
# process((x_i, y_i, z_i))


# %% Loop over all the chunks
coords = itertools.product(
    range(num_chunks_per_dim[0]),
    range(num_chunks_per_dim[1]),
    range(num_chunks_per_dim[2]),
)
# Do it in reverse order because the last chunks are most likely to error
reversed_coords = list(coords)
reversed_coords.reverse()

# %% Move the data across with multiple workers
# TODO because we are using non-aligned writes, we can't use multiple workers
# Or at least, we get a warning about it that I don't think we can ignore
# if we want to use multiple workers, we'd need to fix this to have aligned writes
# but that is trickier because we'd then need to load the data from multiple files
# at the same time
# Because one clean chunk in the data could be informed from multiple files
# in the raw data
# max_workers = 8

# with ProcessPoolExecutor(max_workers=max_workers) as executor:
#     executor.map(process, coords)

# %% Move the data across with a single worker
for coord in reversed_coords:
    process(coord)

# %% Serve the dataset to be used in neuroglancer
vols[0].viewer(port=1337)

# %%
