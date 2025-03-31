# %% Import packages

import itertools
import math
from pathlib import Path
from bfio import BioReader
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import touch, Vec
import json
from neuroglancer.downsample import downsample_with_averaging

# %% Define the path to the files

HERE = Path(__file__).parent

FILEPATH = Path(
    "/media/starfish/Storage/metacell/Isl1-GFP_E13-5_F129-3_CMN-R-L_02052024-GLC-stitched.ome.tiff"
)
OUTPUT_PATH = Path(
    "/media/starfish/Storage/metacell/converted/Isl1-GFP_E13-5_F129-3_CMN-R-L_02052024-GLC-stitched"
)
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
OVERWRITE = False

# %% Load the data
br = BioReader(str(FILEPATH), backend="bioformats")

# %% Inspect the data
with open(OUTPUT_PATH / "metadata.txt", "w") as f:
    json.dump(br.metadata.model_dump_json(), f)
    f.write(br.metadata.model_dump_json())
print(br.shape)

# %% Extract from the data - units are in nm
size_x = br.metadata.images[0].pixels.physical_size_x
if size_x is None:
    size_x = 1
size_y = br.metadata.images[0].pixels.physical_size_y
if size_y is None:
    size_y = 1
size_z = br.metadata.images[0].pixels.physical_size_z
if size_z is None:
    size_z = 1
x_unit = br.metadata.images[0].pixels.physical_size_x_unit
y_unit = br.metadata.images[0].pixels.physical_size_y_unit
z_unit = br.metadata.images[0].pixels.physical_size_z_unit

# if the the units are um, convert to nm
if str(x_unit) == "UnitsLength.MICROMETER":
    size_x *= 1000
if str(y_unit) == "UnitsLength.MICROMETER":
    size_y *= 1000
if str(z_unit) == "UnitsLength.MICROMETER":
    size_z *= 1000

num_channels = br.shape[-1]
data_type = "uint16"
chunk_size = [256, 256, 128]
volume_size = [br.shape[1], br.shape[0], br.shape[2]]  # XYZ

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
    max_mip=2,
    factor=Vec(2, 2, 2),
)
vols = [CloudVolume("file://" + str(OUTPUT_PATH), mip=i) for i in range(3)]
vols[0].provenance.description = "Example data conversion"
vols[0].commit_info()
vols[0].commit_provenance()

# %% Setup somewhere to hold progress
progress_dir = OUTPUT_PATH / "progress"
progress_dir.mkdir(exist_ok=True)


# %% Functions for moving data
shape = np.array([br.shape[1], br.shape[0], br.shape[2]])
chunk_shape = np.array([1024, 1024, 512])  # this is for reading data
num_chunks_per_dim = np.ceil(shape / chunk_shape).astype(int)


def chunked_reader(x_i, y_i, z_i):
    x_start, x_end = x_i * chunk_shape[0], min((x_i + 1) * chunk_shape[0], shape[0])
    y_start, y_end = y_i * chunk_shape[1], min((y_i + 1) * chunk_shape[1], shape[1])
    z_start, z_end = z_i * chunk_shape[2], min((z_i + 1) * chunk_shape[2], shape[2])

    # Read the chunk from the BioReader
    chunk = br.read(X=(x_start, x_end), Y=(y_start, y_end), Z=(z_start, z_end))

    # Return the chunk
    return chunk.swapaxes(0, 1)


def process(args):
    x_i, y_i, z_i = args
    start = [x_i * chunk_shape[0], y_i * chunk_shape[1], z_i * chunk_shape[2]]
    end = [
        min((x_i + 1) * chunk_shape[0], shape[0]),
        min((y_i + 1) * chunk_shape[1], shape[1]),
        min((z_i + 1) * chunk_shape[2], shape[2]),
    ]
    f_name = (
        progress_dir
        / f"{start[0]}-{end[0]}_{start[1]}-{end[1]}_{start[2]}-{end[2]}.done"
    )
    if f_name.exists() and not OVERWRITE:
        return
    print("Working on", f_name)
    rawdata = chunked_reader(x_i, y_i, z_i)
    print(rawdata.shape)
    for mip_level in reversed(range(3)):
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
# max_workers = 8

# with ProcessPoolExecutor(max_workers=max_workers) as executor:
#     executor.map(process, coords)

# %% Move the data across with a single worker
for coord in reversed_coords:
    process(coord)

# %% Serve the dataset to be used in neuroglancer
vols[0].viewer(port=1337)

# %%
