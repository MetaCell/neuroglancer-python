# %% Import packages

import itertools
import math
from pathlib import Path
import time
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
MIP_CUTOFF = 4  # To save time you can start at the lowest resolution and work up

# %% Load the data
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
# Need to figure out some sizes by checking the data
all_files = list(INPUTFOLDER.glob("**/*.zarr"))


def load_zarr_data(file_path):
    zarr_store = zarr.open(file_path, mode="r")
    return zarr_store


def load_zarr_and_permute(file_path):
    zarr_store = zarr.open(file_path, mode="r")
    # Input is in Z, T, C, Y, X order
    # Want XYTCZ order
    data = zarr_store[:]
    data = np.transpose(data, (4, 3, 1, 2, 0))  # Permute to XYTCZ
    return zarr_store, data


def load_chunk_from_zarr_store(
    zarr_store, x_start, x_end, y_start, y_end, z_start, z_end, channel=0
):
    # Input is in Z, T, C, Y, X order
    data = zarr_store[
        :,  # T
        z_start:z_end,  # Z
        channel,  # C
        y_start:y_end,  # Y
        x_start:x_end,  # X
    ]
    # The original timestamp was 65535, can be filterd out
    data = np.where(data == 65535, 0, data)
    print("Loaded data shape b4 sq:", data.shape)
    data = np.squeeze(data)  # Remove any singleton dimensions
    print("Loaded data shape:", data.shape)
    # Then we permute to XYTCZ
    data = np.transpose(data, (-1, -2, 0))  # Permute to XYTCZ
    return data


zarr_store = load_zarr_data(all_files[0])

# It may take too long to just load one file, might need to process in chunks
# %% Check how long to load a single file
start_time = time.time()
data = load_chunk_from_zarr_store(
    zarr_store, 0, 256, 0, 256, 0, 128, channel=0
)
print("Time to load a single file:", time.time() - start_time)

# %% Inspect the data
shape = zarr_store.shape
# Input is in Z, T, C, Y, X order
# Want XYTCZ order
# single_file_shape = [shape[4], shape[3], shape[1], shape[2], shape[0]]
single_file_dims_shape = [shape[4], shape[3], shape[1]]
size_x = 1
size_y = 1
size_z = 1

num_channels = shape[2]
data_type = "uint16"
chunk_size = [256, 256, 128]

# You can provide a subset here also
num_rows = 16
num_cols = 24
volume_size = [
    single_file_dims_shape[0] * num_cols,
    single_file_dims_shape[1] * num_rows,
    single_file_dims_shape[2],
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
)
vol.commit_info()
vol.provenance.description = "Example data conversion"
vol.commit_provenance()
del vol

# %% Create the volumes for each mip level and hold progress
vols = [
    CloudVolume("file://" + str(OUTPUT_PATH), mip=i, compress=False)
    for i in range(NUM_MIPS)
]
progress_dir = OUTPUT_PATH / "progress"
progress_dir.mkdir(exist_ok=True)

# %% Functions for moving data
read_shape = single_file_dims_shape  # this is for reading data


def process(args):
    x_i, y_i = args
    start = [x_i * read_shape[0], y_i * read_shape[1], 0]
    end = [
        (x_i + 1) * read_shape[0],
        (y_i + 1) * read_shape[1],
        read_shape[2],
    ]
    f_name = progress_dir / f"{start[0]}-{end[0]}_{start[1]}-{end[1]}.done"
    if f_name.exists() and not OVERWRITE:
        return
    flat_index = x_i * num_cols + y_i
    path = all_files[flat_index]
    rawdata = load_zarr_and_permute(path)[1]
    print("Working on", f_name)
    for mip_level in reversed(range(MIP_CUTOFF, NUM_MIPS)):
        if mip_level == 0:
            downsampled = rawdata
            ds_start = start
            ds_end = end
        else:
            downsampled = downsample_with_averaging(
                rawdata, [2 * mip_level, 2 * mip_level, 2 * mip_level, 1]
            )
            ds_start = [int(math.ceil(s / (2 * mip_level))) for s in start]
            ds_end = [int(math.ceil(e / (2 * mip_level))) for e in end]

        vols[mip_level][
            ds_start[0] : ds_end[0], ds_start[1] : ds_end[1], ds_start[2] : ds_end[2]
        ] = downsampled
    touch(f_name)


# %% Try with a single chunk to see if it works
x_i, y_i = 0, 0
process((x_i, y_i))

# %% Loop over all the chunks

coords = itertools.product(range(num_rows), range(num_cols))
# Do it in reverse order because the last chunks are most likely to error
reversed_coords = list(coords)
reversed_coords.reverse()

# %% Move the data across with multiple workers
# max_workers = 8

# with ProcessPoolExecutor(max_workers=max_workers) as executor:
#     executor.map(process, coords)

# %% Move the data across with a single worker
for coord in reversed_coords:
    process(coord)

# %% Serve the dataset to be used in neuroglancer
vols[0].viewer(port=1337)

# %%
