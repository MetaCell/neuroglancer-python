# %% Import packages

import itertools
import math
from pathlib import Path
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import touch, Vec
import zarr
from neuroglancer.downsample import downsample_with_averaging
from google.cloud import storage
import os

# Try to import dotenv, fall back to manual parsing if not available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print("Warning: python-dotenv not installed. Install with 'pip install python-dotenv' for .env file support.")

# %% Load environment configuration

HERE = Path(__file__).parent

def parse_bool(value):
    """Parse string boolean values to Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on')

def load_env_config():
    """
    Load configuration from environment variables.
    First tries to load from .env file if python-dotenv is available,
    then falls back to system environment variables.
    """
    # Load .env file if dotenv is available
    env_file = HERE / '.env'
    if HAS_DOTENV and env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded configuration from {env_file}")
    elif env_file.exists():
        print(f"Found {env_file} but python-dotenv not available. Using system environment variables only.")
    else:
        print("No .env file found. Using system environment variables only.")
    
    # Load configuration from environment variables with defaults
    config = {
        # Data source configuration
        'USE_GCS_BUCKET': parse_bool(os.getenv('USE_GCS_BUCKET', 'false')),
        'GCS_BUCKET_NAME': os.getenv('GCS_BUCKET_NAME', 'your-bucket-name'),
        'GCS_PREFIX': os.getenv('GCS_PREFIX', 'path/to/zarr/files/'),
        'GCS_FILE_EXTENSION': os.getenv('GCS_FILE_EXTENSION', '.zarr'),
        
        # Local paths (used when USE_GCS_BUCKET is False)
        'INPUTFOLDER': Path(os.getenv('INPUTFOLDER', '/temp/in')),
        'OUTPUT_PATH': Path(os.getenv('OUTPUT_PATH', '/temp/out')),
        
        # Processing settings
        'OVERWRITE': parse_bool(os.getenv('OVERWRITE', 'false')),
        'NUM_MIPS': int(os.getenv('NUM_MIPS', '5')),
        'MIP_CUTOFF': int(os.getenv('MIP_CUTOFF', '0')),
        'CHANNEL_LIMIT': int(os.getenv('CHANNEL_LIMIT', '4')),
        'NUM_ROWS': int(os.getenv('NUM_ROWS', '3')),
        'NUM_COLS': int(os.getenv('NUM_COLS', '6')),
        'ALLOW_NON_ALIGNED_WRITE': parse_bool(os.getenv('ALLOW_NON_ALIGNED_WRITE', 'false')),
        
        # Optional resolution settings
        'SIZE_X': int(os.getenv('SIZE_X', '1')),
        'SIZE_Y': int(os.getenv('SIZE_Y', '1')),
        'SIZE_Z': int(os.getenv('SIZE_Z', '1')),
        
        # Optional chunk settings
        'CHUNK_SIZE_X': int(os.getenv('CHUNK_SIZE_X', '64')),
        'CHUNK_SIZE_Y': int(os.getenv('CHUNK_SIZE_Y', '64')),
        'CHUNK_SIZE_Z': int(os.getenv('CHUNK_SIZE_Z', '32')),
    }
    
    return config

# Load configuration
config = load_env_config()

# Extract configuration variables for backward compatibility
USE_GCS_BUCKET = config['USE_GCS_BUCKET']
GCS_BUCKET_NAME = config['GCS_BUCKET_NAME']
GCS_PREFIX = config['GCS_PREFIX']
GCS_FILE_EXTENSION = config['GCS_FILE_EXTENSION']
INPUTFOLDER = config['INPUTFOLDER']
OUTPUT_PATH = config['OUTPUT_PATH']
OVERWRITE = config['OVERWRITE']
NUM_MIPS = config['NUM_MIPS']
MIP_CUTOFF = config['MIP_CUTOFF']
CHANNEL_LIMIT = config['CHANNEL_LIMIT']
NUM_ROWS = config['NUM_ROWS']
NUM_COLS = config['NUM_COLS']
ALLOW_NON_ALIGNED_WRITE = config['ALLOW_NON_ALIGNED_WRITE']

# Print loaded configuration for verification
print("Configuration loaded:")
print(f"  Data source: {'GCS Bucket' if USE_GCS_BUCKET else 'Local files'}")
if USE_GCS_BUCKET:
    print(f"  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"  GCS Prefix: {GCS_PREFIX}")
else:
    print(f"  Input folder: {INPUTFOLDER}")
print(f"  Output path: {OUTPUT_PATH}")
print(f"  Processing: {NUM_MIPS} mips, {CHANNEL_LIMIT} channels, {NUM_ROWS}x{NUM_COLS} grid")

# %% Load the data
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

def list_gcs_files(bucket_name, prefix="", file_extension=""):
    """
    List files from a Google Cloud Storage bucket that match the given prefix and extension.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix path within the bucket to filter files
        file_extension: File extension to filter for (e.g., '.zarr')
        
    Returns:
        List of GCS blob names that match the criteria
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    filtered_files = []
    
    for blob in blobs:
        if blob.name.endswith(file_extension):
            filtered_files.append(blob.name)
    
    print(f"Found {len(filtered_files)} files matching '{file_extension}' extension in bucket '{bucket_name}' with prefix '{prefix}'")
    return filtered_files

def get_file_list():
    """
    Get the list of files either from GCS bucket or local filesystem based on configuration.
    
    Returns:
        List of file paths (GCS blob names or local Path objects)
    """
    if USE_GCS_BUCKET:
        files = list_gcs_files(GCS_BUCKET_NAME, GCS_PREFIX, GCS_FILE_EXTENSION)
    else:
        # Use local filesystem glob as before
        files = list(INPUTFOLDER.glob(f"**/*{GCS_FILE_EXTENSION}"))
    print(f"Total files found: {len(files)}")
    return sorted(files)

# Get the list of available files
all_files = get_file_list()


# TODO change this to be more robust.
# It needs to have three things.
# 1. It needs to be able to convert a row, col to the remote google cloud bucket
# and the corresponding local file path for that file to be downloaded to
# 2. It needs to be able to download the file if it is not already present
# 3. It needs to be able to delete the file after use

# TODO make num rows and num cols be inferred from the files present
# by looking at the file names and seeing what the max row and col are
# instead of being hard coded
def get_file_for_row_col(row, col):
    """Get the file path for a specific row and column."""
    if row < 0 or row >= NUM_ROWS or col < 0 or col >= NUM_COLS:
        raise ValueError("Row and column indices must be within the defined grid.")
    index = row * NUM_COLS + col
    return all_files[index] if index < len(all_files) else None


def load_zarr_store(file_path):
    zarr_store = zarr.open(file_path, mode="r")
    return zarr_store


def load_data_from_zarr_store(zarr_store):
    # Input is in Z, T, C, Y, X order
    # TODO compute the number of channels by the min of the number of channels
    # and the hard coded channel upper bound limit
    data = zarr_store[
        :,  # T
        :,  # Z
        :CHANNEL_LIMIT,  # C
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
size_x = config['SIZE_X']
size_y = config['SIZE_Y']
size_z = config['SIZE_Z']
# Here, T and Z are kind of transferrable.
# The reason is because the z dimension in neuroglancer is the time dimension
# from the raw original data.
# So both terms might be used to represent the same thing.
# It's a bit unusual that t is being used as the z dimension,
# but otherwise you can't do volume rendering in neuroglancer.

num_channels = min(shape[2], CHANNEL_LIMIT)  # Limit to NUM_CHANNELS for memory usage
data_type = "uint16"

# TODO compute the chunk size based on the single file xyz shape
# and the number of mips, so that the chunks exactly line up
# with the octree structure of the data
chunk_size = [config['CHUNK_SIZE_X'], config['CHUNK_SIZE_Y'], config['CHUNK_SIZE_Z']]  # chunk size in neuroglancer
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

# TODO we need to sync the output path info file with the google cloud bucket
# so that the bucket is ready to receive the rest of the data
# after the cloudvolume is created

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
    # TODO we need to here do something a little bit complex
    # we need to check based on which co-ordinates we have done
    # whether any of the output files are fully written
    # if they are, we can upload them to the google cloud bucket
    # and then delete them locally to save space
    # This is a bit complex because we need to check across all mips
    # and see if any of the files are fully written
    # and then upload them
    # This is important because otherwise we will run out of space
    # on the local disk

# %% Serve the dataset to be used in neuroglancer
vols[0].viewer(port=1337)

# %%
