# %% Import packages

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

FILEPATH = Path(
    "..."
)
OUTPUT_PATH = Path(
    "..."
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
    # TEMP
    print(np.unique(rawdata))
    print(rawdata.shape)
    # Create the segmentation mask
    dimensions = [start, end]
    seg_chunk = create_segmentation_chunk(rawdata, dimensions, convert_non_zero_to=None)
    seg_chunk.write_to_directory(OUTPUT_PATH / "data")
    touch(f_name)


# %% Try with a single chunk to see if it works
x_i, y_i, z_i = 0, 0, 0
process((x_i, y_i, z_i))

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
metadata = {
    "@type": "neuroglancer_multiscale_volume",
    "data_type": "uint32",
    "num_channels": 1,
    "scales": [
        {
            "chunk_sizes": [chunk_size],
            "encoding": "compressed_segmentation",
            "copmressed_segmentation_block_size": [8, 8, 8],
            "resolution": [size_x, size_y, size_z],
            "key": "data",
            "size": volume_size,
        }
    ],
    "mesh": "mesh",
    "type": "segmentation",
}
write_metadata(
    metadata,
    OUTPUT_PATH,
    overwrite=OVERWRITE,
)

# %% Now create the mesh
mesh_shape = np.array(volume_size)
generate_multiresolution_mesh_from_segmentation(OUTPUT_PATH, "mesh", 3, mesh_shape)


# %% Move the data across with a single worker
for coord in reversed_coords:
    process(coord)
