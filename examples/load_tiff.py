# %% Import packages

import itertools
from pathlib import Path
from bfio import BioReader
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import touch
import json

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
chunk_size = [256, 256, 128, 1]
volume_size = [br.shape[1], br.shape[0], br.shape[2]] # XYZ

# %% Setup the cloudvolume info
info = CloudVolume.create_new_info(
    num_channels=num_channels,
    layer_type="image",
    data_type=data_type,
    encoding="raw",
    resolution=[size_x, size_y, size_z],
    voxel_offset=[0, 0, 0],
    chunk_size=chunk_size[:-1],
    volume_size=volume_size,
)
vol = CloudVolume("file://" + str(OUTPUT_PATH), info=info)
vol.provenance.description = "Example data conversion"
vol.commit_info()
vol.commit_provenance()

# %% Setup somewhere to hold progress
progress_dir = OUTPUT_PATH / "progress"
progress_dir.mkdir(exist_ok=True)


# %% Functions for moving data
shape = np.array([br.shape[1], br.shape[0], br.shape[2], br.shape[3]])
chunk_shape = np.array([1024, 1024, 512, 1])  # this is for reading data
num_chunks_per_dim = np.ceil(shape / chunk_shape).astype(int)


def chunked_reader(x_i, y_i, z_i, c):
    x_start, x_end = x_i * chunk_shape[0], min((x_i + 1) * chunk_shape[0], shape[0])
    y_start, y_end = y_i * chunk_shape[1], min((y_i + 1) * chunk_shape[1], shape[1])
    z_start, z_end = z_i * chunk_shape[2], min((z_i + 1) * chunk_shape[2], shape[2])

    # Read the chunk from the BioReader
    chunk = br.read(
        X=(x_start, x_end), Y=(y_start, y_end), Z=(z_start, z_end), C=(c,)
    )
    # Keep expanding dims until it is the same length as chunk_shape
    while len(chunk.shape) < len(chunk_shape):
        chunk = np.expand_dims(chunk, axis=-1)
    # Return the chunk
    return chunk.swapaxes(0, 1)


def process(args):
    x_i, y_i, z_i, c = args
    start = [x_i * chunk_shape[0], y_i * chunk_shape[1], z_i * chunk_shape[2]]
    end = [
        min((x_i + 1) * chunk_shape[0], shape[0]),
        min((y_i + 1) * chunk_shape[1], shape[1]),
        min((z_i + 1) * chunk_shape[2], shape[2]),
    ]
    f_name = (
        progress_dir
        / f"{start[0]}-{end[0]}_{start[1]}-{end[1]}_{start[2]}-{end[2]}_{c}.done"
    )
    if f_name.exists() and not OVERWRITE:
        return
    print("Working on", f_name)
    rawdata = chunk = chunked_reader(x_i, y_i, z_i, c)
    vol[start[0] : end[0], start[1] : end[1], start[2] : end[2], c] = rawdata
    touch(f_name)


# %% Try with a single chunk to see if it works
# x_i, y_i, z_i = 0, 0, 0
# process((x_i, y_i, z_i, 0))

# %% Can't figure out the writing so do it with fake data
# fake_data = np.random.randint(0, 2**16, size=chunk_size, dtype=np.uint16)
# vol[0:256, 0:256, 0:128, 0] = fake_data

coords = itertools.product(
    range(num_chunks_per_dim[0]),
    range(num_chunks_per_dim[1]),
    range(num_chunks_per_dim[2]),
    range(num_channels),
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
vol.viewer(port=1337)
