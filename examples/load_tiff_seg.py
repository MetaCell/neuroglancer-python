# %% Import packages

from copy import copy
import itertools
from pathlib import Path
from bfio import BioReader
import numpy as np
from cloudvolume.lib import touch
import json
from cryoet_data_portal_neuroglancer.precompute.segmentation_mask import (
    create_segmentation_chunk,
    write_metadata,
)
from cryoet_data_portal_neuroglancer.precompute.mesh import (
    generate_multiresolution_mesh_from_segmentation,
)

# %% Define the path to the files

HERE = Path(__file__).parent

FILEPATH = Path("...")
OUTPUT_PATH = Path("...")
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
volume_size = [br.shape[1], br.shape[0], br.shape[2]]  # XYZ

# %% Setup somewhere to hold progress
CHANNEL = 0
progress_dir = OUTPUT_PATH / "progress"

# %% Functions for moving data
shape = np.array([br.shape[1], br.shape[0], br.shape[2]])
chunk_shape = np.array([1024, 1024, 512])  # this is for reading data
num_chunks_per_dim = np.ceil(shape / chunk_shape).astype(int)


def chunked_reader(x_i, y_i, z_i):
    x_start, x_end = x_i * chunk_shape[0], min((x_i + 1) * chunk_shape[0], shape[0])
    y_start, y_end = y_i * chunk_shape[1], min((y_i + 1) * chunk_shape[1], shape[1])
    z_start, z_end = z_i * chunk_shape[2], min((z_i + 1) * chunk_shape[2], shape[2])

    # Read the chunk from the BioReader
    chunk = br.read(
        X=(x_start, x_end), Y=(y_start, y_end), Z=(z_start, z_end), C=(CHANNEL,)
    )
    chunk = np.atleast_3d(chunk)
    # Remove the last flattened dimension
    if (len(chunk.shape) > 3) and (chunk.shape[-1] == 1):
        chunk = chunk[:, :, :, 0]

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
    rawdata = chunked_reader(x_i, y_i, z_i)
    # TEMP
    print(rawdata.shape)
    # Create the segmentation mask
    # The writer expects the data to be in ZYX order so need to swap the axes
    start_zyx = [start[2], start[1], start[0]]
    end_zyx = [end[2], end[1], end[0]]
    dimensions = [start_zyx, end_zyx]
    seg_chunk = create_segmentation_chunk(rawdata, dimensions, convert_non_zero_to=1)
    seg_chunk.write_to_directory(OUTPUT_PATH / "data")
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

# %% Create the metadata
int_size = [int(size_x), int(size_y), int(size_z)]
real_size = [size_x, size_y, size_z]
metadata = {
    "@type": "neuroglancer_multiscale_volume",
    "data_type": "uint32",
    "num_channels": 1,
    "scales": [
        {
            "chunk_sizes": [[int(c) for c in chunk_shape]],
            "encoding": "compressed_segmentation",
            "compressed_segmentation_block_size": [8, 8, 8],
            "resolution": int_size,
            "key": "data",
            "size": volume_size,
        }
    ],
    "mesh": "mesh",
    "type": "segmentation",
}

# %% Move the data across with a single worker
original_path = copy(OUTPUT_PATH)
for i in range(0, 1):
    OUTPUT_PATH = original_path.with_stem(f"{original_path.stem}_ch{i}")
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    print("Writing to ", OUTPUT_PATH)
    # Make the progress directory
    progress_dir = OUTPUT_PATH / "progress"
    progress_dir.mkdir(exist_ok=True)
    CHANNEL = i
    write_metadata(metadata, OUTPUT_PATH)
    for coord in reversed_coords:
        process(coord)
        print("COMPLETED", i)
    mesh_shape = np.array(volume_size)
    generate_multiresolution_mesh_from_segmentation(
        OUTPUT_PATH, "mesh", 2, mesh_shape, fill_missing=True
    )

# %% For running all
