# %% Import packages

import itertools
import math
import subprocess
from pathlib import Path
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import touch, Vec
import zarr
from neuroglancer.downsample import downsample_with_averaging
from google.cloud import storage
import os
import re

# Try to import dotenv, fall back to manual parsing if not available
try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print(
        "Warning: python-dotenv not installed. Install with 'pip install python-dotenv' for .env file support."
    )


# %% Load environment configuration

HERE = Path(__file__).parent


def parse_bool(value):
    """Parse string boolean values to Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "on")


def load_env_config():
    """
    Load configuration from environment variables.
    First tries to load from .env file if python-dotenv is available,
    then falls back to system environment variables.
    """
    # Load .env file if dotenv is available
    env_file = HERE / ".env"
    if HAS_DOTENV and env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded configuration from {env_file}")
    elif env_file.exists():
        print(
            f"Found {env_file} but python-dotenv not available. Using system environment variables only."
        )
    else:
        print("No .env file found. Using system environment variables only.")

    # Load configuration from environment variables with defaults
    config = {
        # Data source configuration
        "USE_GCS_BUCKET": parse_bool(os.getenv("USE_GCS_BUCKET", "false")),
        "GCS_BUCKET_NAME": os.getenv("GCS_BUCKET_NAME", "your-bucket-name"),
        "GCS_PREFIX": os.getenv("GCS_PREFIX", "path/to/zarr/files/"),
        "GCS_FILE_EXTENSION": os.getenv("GCS_FILE_EXTENSION", ".zarr"),
        "GCS_FILES_LOCAL_LIST": Path(os.getenv("GCS_FILES_LOCAL_LIST", "")),
        "GCS_PROJECT": os.getenv("GCS_PROJECT", None),
        # Output GCS bucket configuration (for uploading results)
        "USE_GCS_OUTPUT": parse_bool(os.getenv("USE_GCS_OUTPUT", "false")),
        "GCS_OUTPUT_BUCKET_NAME": os.getenv(
            "GCS_OUTPUT_BUCKET_NAME", "your-output-bucket-name"
        ),
        "GCS_OUTPUT_PREFIX": os.getenv("GCS_OUTPUT_PREFIX", "processed/"),
        # Local paths (used when USE_GCS_BUCKET is False)
        "INPUT_PATH": Path(os.getenv("INPUT_PATH", "/temp/in")),
        "OUTPUT_PATH": Path(os.getenv("OUTPUT_PATH", "/temp/out")),
        "DELETE_INPUT": parse_bool(os.getenv("DELETE_INPUT", "false")),
        "DELETE_OUTPUT": parse_bool(os.getenv("DELETE_OUTPUT", "false")),
        # Processing settings
        "OVERWRITE": parse_bool(os.getenv("OVERWRITE", "false")),
        "OVERWRITE_GCS": parse_bool(os.getenv("OVERWRITE_GCS", "false")),
        "NUM_MIPS": int(os.getenv("NUM_MIPS", "5")),
        "MIP_CUTOFF": int(os.getenv("MIP_CUTOFF", "0")),
        "CHANNEL_LIMIT": int(os.getenv("CHANNEL_LIMIT", "4")),
        "ALLOW_NON_ALIGNED_WRITE": parse_bool(
            os.getenv("ALLOW_NON_ALIGNED_WRITE", "false")
        ),
        # Process possible comma separated list of integers for manual chunk size
        "MANUAL_CHUNK_SIZE": (
            [int(x) for x in str(os.getenv("MANUAL_CHUNK_SIZE", "None")).split(",")]
            if os.getenv("MANUAL_CHUNK_SIZE", "None").lower() != "none"
            else None
        ),
    }

    return config


# Load configuration
config = load_env_config()

# Extract configuration variables for backward compatibility
use_gcs_bucket = config["USE_GCS_BUCKET"]
gcs_bucket_name = config["GCS_BUCKET_NAME"]
gcs_input_path = config["GCS_PREFIX"]
gcs_file_ext = config["GCS_FILE_EXTENSION"]
gcs_local_list = config["GCS_FILES_LOCAL_LIST"]
gcs_project = config["GCS_PROJECT"]
use_gcs_output = config["USE_GCS_OUTPUT"]
gcs_output_bucket_name = config["GCS_OUTPUT_BUCKET_NAME"]
gcs_output_path = config["GCS_OUTPUT_PREFIX"]
input_path = config["INPUT_PATH"]
output_path = config["OUTPUT_PATH"]
delete_input = config["DELETE_INPUT"]
delete_output = config["DELETE_OUTPUT"]
overwrite_output = config["OVERWRITE"]
overwrite_gcs = config["OVERWRITE_GCS"]
num_mips = config["NUM_MIPS"]
mip_cutoff = config["MIP_CUTOFF"]
channel_limit = config["CHANNEL_LIMIT"]
allow_non_aligned_write = config["ALLOW_NON_ALIGNED_WRITE"]
manual_chunk_size = config["MANUAL_CHUNK_SIZE"]

# Print loaded configuration for verification
print("Configuration loaded:")
print(f"  Data source: {'GCS Bucket' if use_gcs_bucket else 'Local files'}")
if use_gcs_bucket:
    print(f"  GCS Bucket: {gcs_bucket_name}")
    print(f"  GCS Prefix: {gcs_input_path}")
else:
    print(f"  Input folder: {input_path}")
print(f"  Output path: {output_path}")
print(f"  Output to GCS: {'Yes' if use_gcs_output else 'No'}")
if use_gcs_output:
    print(f"  Output GCS Bucket: {gcs_output_bucket_name}")
    print(f"  Output GCS Prefix: {gcs_output_path}")
print(f"  Processing: {num_mips} mips, max {channel_limit} channels")

# %% Load the list of files to process
output_path.mkdir(exist_ok=True, parents=True)


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
    if gcs_local_list and gcs_local_list.exists():
        print(f"Loading file list from local file: {gcs_local_list}")
        with open(gcs_local_list, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Found {len(files)} files in local list")
        return files
    client = storage.Client(project=gcs_project)
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    filtered_files = []

    for blob in blobs:
        if blob.name.endswith(file_extension):
            filtered_files.append(blob.name)

    print(
        f"Found {len(filtered_files)} files matching '{file_extension}' extension in bucket '{bucket_name}' with prefix '{prefix}'"
    )
    return filtered_files


def get_file_list():
    """
    Get the list of files either from GCS bucket or local filesystem based on configuration.

    Returns:
        List of file paths (GCS blob names or local Path objects)
    """
    if use_gcs_bucket:
        files = list_gcs_files(gcs_bucket_name, gcs_input_path, gcs_file_ext)
    else:
        # Use local filesystem glob as before
        files = list(input_path.glob(f"**/*{gcs_file_ext}"))
    print(f"Total files found: {len(files)}")
    return sorted(files)


# Get the list of available files
all_files = get_file_list()

# Write the files out to a text file for reference
with open(output_path / "file_list.txt", "w") as f:
    for filepath in all_files:
        f.write(f"{filepath}\n")

# %%

all_files

# %% Determine grid dimensions from files


def extract_row_col_from_filename(filename):
    """
    Extract row and column information from filename.
    Assumes filenames contain row/col info in a pattern like 'r{row}c{col}' or '{row}_{col}'.

    Args:
        filename: The filename to parse

    Returns:
        tuple: (row, col) or (None, None) if pattern not found
    """
    filename_str = str(filename)

    # Try pattern like 'r0c1' or 'row0col1'
    match = re.search(r"r(?:ow)?(\d+)c(?:ol)?(\d+)", filename_str, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try pattern like '0_1' or '00_01'
    match = re.search(r"(\d+)_(\d+)", filename_str)
    if match:
        return int(match.group(1)), int(match.group(2))

    raise ValueError(f"Could not extract row/col from filename: {filename_str}")


def compute_grid_dimensions(file_list):
    """
    Compute the number of rows and columns from the available files.

    Args:
        file_list: List of file paths or names

    Returns:
        tuple: (num_rows, num_cols)
    """
    if not file_list:
        raise ValueError("File list is empty, cannot compute grid dimensions.")

    max_row = -1
    max_col = -1
    for filepath in file_list:
        row, col = extract_row_col_from_filename(filepath)
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    return max_row + 1, max_col + 1  # Convert from max index to count


# Compute actual grid dimensions from files
computed_num_rows, computed_num_cols = compute_grid_dimensions(all_files)
print(
    f"Using computed grid dimensions: {computed_num_rows} rows x {computed_num_cols} columns"
)

# %% GCS Output Functions and loading data functions


def get_local_cache_path(row, col):
    """
    Get the local cache path where a file for the given row/col should be stored.

    Args:
        row: Row index
        col: Column index

    Returns:
        Path: Local cache path for the file
    """

    # Get the remote file path/name for this row/col
    remote_file = get_remote_file_path(row, col)
    if remote_file is None:
        return None

    # Create local filename based on remote file
    if use_gcs_bucket:
        cache_dir = input_path
        cache_dir.mkdir(exist_ok=True, parents=True)
        local_name = str(remote_file).rstrip("/").split("/")[-1]
        output = Path(cache_dir / local_name)
    else:
        output = Path(remote_file)

    return output


def get_remote_file_path(row, col):
    """
    Get the remote file path (GCS blob name or local path) for a specific row and column.

    Args:
        row: Row index
        col: Column index

    Returns:
        str or Path: Remote file path, or None if not found
    """
    if row < 0 or row >= computed_num_rows or col < 0 or col >= computed_num_cols:
        raise ValueError(
            f"Row and column indices must be within the defined grid (0-{computed_num_rows-1}, 0-{computed_num_cols-1})."
        )

    # Try to find file by row/col pattern first
    for filepath in all_files:
        file_row, file_col = extract_row_col_from_filename(filepath)
        if file_row == row and file_col == col:
            return filepath

    return None


def gcloud_download_dir(gs_prefix: str, local_dir: Path) -> None:
    """
    Recursively download a GCS prefix to a local directory using gcloud.
    Example gs_prefix: 'gs://my-bucket/some/prefix/'
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # Use a list (no shell=True) to avoid injection issues
    cmd = [
        "gcloud",
        "storage",
        "cp",
        "--recursive",
        "--project",
        gcs_project,
        gs_prefix,
        str(local_dir),
    ]

    print("Running command:", " ".join(cmd))
    try:
        res = subprocess.run(
            cmd,
            check=True,  # raises CalledProcessError on nonzero exit
            capture_output=True,  # capture logs; integrate with your logger
            text=True,
        )
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
    except subprocess.CalledProcessError as e:
        # Surface meaningful diagnostics
        print("gcloud cp failed:", e.returncode)
        print(e.stdout)
        print(e.stderr)
        raise


def download_zarr_file(row, col):
    """
    Download the file for a specific row and column to local cache.

    Args:
        row: Row index
        col: Column index

    Returns:
        Path: Local cache path of downloaded file, or None if not found
    """
    remote_file = get_remote_file_path(row, col)
    if remote_file is None:
        print(f"No file found for row {row}, col {col}")
        return None

    local_path = get_local_cache_path(row, col)
    if local_path is None:
        return None

    # If file already exists locally, no need to download
    if local_path.exists():
        print(f"File already cached: {local_path}")
        return local_path

    local_path.mkdir(exist_ok=True, parents=True)

    if use_gcs_bucket:
        gcloud_download_dir(remote_file, local_path.parent)
        return local_path
    else:
        return remote_file  # For local files, just return the path


def load_file(row, col):
    """
    Load the zarr store for a specific row and column.
    Downloads the file first if not already cached locally.

    Args:
        row: Row index
        col: Column index

    Returns:
        zarr store object, or None if not found/error
    """
    local_path = download_zarr_file(row, col)
    if local_path is None:
        return None

    try:
        print(f"Loading zarr store from {local_path}")
        zarr_store = zarr.open(str(local_path), mode="r")
        return zarr_store
    except Exception as e:
        print(f"Error loading zarr store from {local_path}: {e}")
        return None


def delete_cached_zarr_file(row, col):
    """
    Delete the locally cached file for a specific row and column to save disk space.

    Args:
        row: Row index
        col: Column index

    Returns:
        bool: True if file was deleted or didn't exist, False if error
    """
    if not use_gcs_bucket or not delete_input:
        return True
    local_path = get_local_cache_path(row, col)
    if local_path is None:
        return True

    try:
        # Check that local_path is not something dangerous like root or home directory
        if local_path in [Path("/"), Path.home()]:
            print(f"Refusing to delete dangerous path: {local_path}")
            return False
        # It should also end with .zarr
        if not local_path.suffix == ".zarr":
            print(f"Refusing to delete non-zarr path: {local_path}")
            return False
        if local_path.exists():
            local_path.rmdir()
            print(f"Deleted cached file: {local_path}")
        return True
    except Exception as e:
        print(f"Error deleting cached file {local_path}: {e}")
        return False


def sync_info_to_gcs_output():
    """
    Sync the CloudVolume info file to the GCS output bucket.
    This uploads the info file so the bucket is ready to receive the rest of the data.
    """
    local_info_path = output_path / "info"
    gcs_info_path = gcs_output_path.rstrip("/") + "/info"
    upload_file_to_gcs(local_info_path, gcs_info_path)


def upload_file_to_gcs(local_file_path, gcs_file_path, overwrite=True):
    """
    Upload a single chunk file to the GCS output bucket.

    Args:
        local_file_path: Path to local chunk file
        gcs_file_path: GCS blob path for the chunk
        overwrite: If False, skip upload if file already exists in GCS

    Returns:
        bool: True if successful, False otherwise
    """
    if not use_gcs_output:
        return True

    try:
        client = storage.Client(project=gcs_project)
        bucket = client.bucket(gcs_output_bucket_name)

        blob = bucket.blob(gcs_file_path)

        # Check if file already exists and overwrite is False
        if not overwrite and blob.exists():
            print(f"File {gcs_file_path} already exists in GCS, skipping upload")
            return True

        blob.upload_from_filename(str(local_file_path))

        return True

    except Exception as e:
        print(f"Error uploading chunk {local_file_path} to GCS: {e}")
        return False


def load_zarr_store(file_path):
    """
    Load zarr store from a file path.
    This function is kept for backward compatibility.
    """
    zarr_store = zarr.open(str(file_path), mode="r")
    return zarr_store


def load_data_from_zarr_store(zarr_store):
    # Input is in Z, T, C, Y, X order

    data = zarr_store[
        :,  # T
        :,  # Z
        :num_channels,  # C
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


# %% Inspect the first file to determine data properties - for now we assume all files are the same shape

# Load the first file to inspect data shape and properties
print("Loading first file for data inspection...")
zarr_store = load_file(0, 0)  # Load file at row 0, col 0
if zarr_store is None:
    print("Error: Could not load first file for data inspection")
    exit(1)

# %% Inspect the data
volume_shape = zarr_store.shape
# Input is in Z, T, C, Y, X order
# Want XYTCZ order
single_file_xyz_shape = [volume_shape[4], volume_shape[3], volume_shape[1]]
# Here, T and Z are kind of transferrable.
# The reason is because the z dimension in neuroglancer is the time dimension
# from the raw original data.
# So both terms might be used to represent the same thing.
# It's a bit unusual that t is being used as the z dimension,
# but otherwise you can't do volume rendering in neuroglancer.

num_channels = min(
    volume_shape[2], channel_limit
)  # Limit to NUM_CHANNELS for memory usage
data_type = "uint16"

# %% Compute optimal chunk size based on data shape and MIP levels


def compute_optimal_chunk_size(single_file_shape, num_mips, max_chunk_size=None):
    """
    Compute optimal chunk size based on single file shape and number of MIP levels.

    Args:
        single_file_shape: [x, y, z] shape of a single file
        num_mips: Number of MIP levels
        max_chunk_size: Optional maximum chunk size (default: 512)

    Returns:
        List[int]: [chunk_x, chunk_y, chunk_z] optimal chunk sizes
    """
    if max_chunk_size is None:
        max_chunk_size = 512

    single_file_shape = np.array(single_file_shape)
    optimal_chunks = np.ceil(single_file_shape / (2 ** (num_mips - 1)))

    return [int(c) for c in optimal_chunks]


# Compute optimal chunk size based on single file shape and MIP levels
print(
    f"Computing optimal chunk size for shape {single_file_xyz_shape} with {num_mips} MIP levels..."
)
computed_chunk_size = compute_optimal_chunk_size(single_file_xyz_shape, num_mips)
if manual_chunk_size is not None:
    if len(manual_chunk_size) != 3:
        print(
            "Error: MANUAL_CHUNK_SIZE must be a list of three integers (e.g., 64,64,16)"
        )
        exit(1)
    print(f"Using manual chunk size from configuration: {manual_chunk_size}")
    chunk_size = manual_chunk_size
else:
    computed_chunk_size = compute_optimal_chunk_size(single_file_xyz_shape, num_mips)
    print(f"Computed optimal chunk size: {computed_chunk_size}")
    chunk_size = computed_chunk_size

volume_size = [
    single_file_xyz_shape[0] * computed_num_rows,
    single_file_xyz_shape[1] * computed_num_cols,
    single_file_xyz_shape[2],
]  # XYZ (T)
print("Volume size:", volume_size)

# Validate chunk size works with the data
for i, (dim_name, dim_size, chunk_dim) in enumerate(
    zip(["X", "Y", "Z"], volume_size, chunk_size)
):
    num_chunks_this_dim = math.ceil(dim_size / chunk_dim)
    print(
        f"  {dim_name} dimension: {dim_size} → {num_chunks_this_dim} chunks of size {chunk_dim}"
    )

    # Check how this works across MIP levels
    for mip in range(num_mips):  # Show first few MIP levels
        effective_size = dim_size // (2**mip)
        if effective_size > 0:
            mip_chunks = math.ceil(effective_size / chunk_dim)
            utilization = (
                (effective_size / (mip_chunks * chunk_dim)) * 100
                if mip_chunks > 0
                else 0
            )
            print(
                f"    MIP {mip}: {effective_size} → {mip_chunks} chunks ({utilization:.1f}% utilization)"
            )
# The chunk size remains fixed across all mips, but at higher mips
# the data will be downsampled, so the effective chunk size will be larger.
# Because we use precomputed data format, every chunk has to have all channels included.
# The optimal chunk size balances:
# - Memory usage (smaller chunks use less memory)
# - I/O efficiency (larger chunks reduce overhead)
# - Octree alignment (powers of 2 work best)
# - Downsampling efficiency (should divide well at all MIP levels)

# %% Setup the cloudvolume info
info = CloudVolume.create_new_info(
    num_channels=num_channels,
    layer_type="image",
    data_type=data_type,
    encoding="raw",
    resolution=[1, 1, 1],
    voxel_offset=[0, 0, 0],
    chunk_size=chunk_size,
    volume_size=volume_size,
    max_mip=num_mips - 1,
    factor=Vec(2, 2, 2),
)
vol = CloudVolume(
    "file://" + str(output_path),
    info=info,
    mip=0,
    non_aligned_writes=allow_non_aligned_write,
    fill_missing=True,
)
vol.commit_info()
vol.provenance.description = "Example data conversion"
vol.commit_provenance()

# Sync the info file to GCS output bucket if configured
sync_info_to_gcs_output()

del vol

# %% Create the volumes for each mip level and hold progress
vols = [
    CloudVolume(
        "file://" + str(output_path),
        mip=i,
        compress=False,
        non_aligned_writes=allow_non_aligned_write,
        fill_missing=True,
    )
    for i in range(num_mips)
]
progress_dir = output_path / "progress"
progress_dir.mkdir(exist_ok=True)

# %% Functions for moving data
volume_shape = volume_size
single_file_shape = np.array(single_file_xyz_shape)  # this is for reading data
num_chunks_per_dim = np.ceil(volume_shape / single_file_shape).astype(int)


def process(args):
    x_i, y_i, z_i = args

    start = [
        x_i * single_file_shape[0],
        y_i * single_file_shape[1],
        z_i * single_file_shape[2],
    ]
    end = [
        min((x_i + 1) * single_file_shape[0], volume_shape[0]),
        min((y_i + 1) * single_file_shape[1], volume_shape[1]),
        min((z_i + 1) * single_file_shape[2], volume_shape[2]),
    ]
    f_name = progress_dir / f"{start[0]}-{end[0]}_{start[1]}-{end[1]}.done"
    print(f"Processing chunk: {start} to {end}, file: {f_name}")
    if f_name.exists() and not overwrite_output:
        return (start, end)

    # Use the new load_file function that handles download/caching
    print(f"Loading file for coordinates ({x_i}, {y_i}, {z_i})")
    loaded_zarr_store = load_file(x_i, y_i)

    if loaded_zarr_store is None:
        print(f"Warning: Could not load file for row {x_i}, col {y_i}. Skipping...")
        return

    rawdata = load_data_from_zarr_store(loaded_zarr_store)

    # Process all mip levels
    for mip_level in reversed(range(mip_cutoff, num_mips)):
        if mip_level == 0:
            downsampled = rawdata
            ds_start = start
            ds_end = end
            if not allow_non_aligned_write:
                # Align to chunk boundaries
                ds_start = [
                    int(round(math.floor(s / c) * c))
                    for s, c in zip(ds_start, chunk_size)
                ]
                ds_end = [
                    int(round(math.ceil(e / c) * c)) for e, c in zip(ds_end, chunk_size)
                ]
                ds_end = [min(e, s) for e, s in zip(ds_end, volume_shape)]
        else:
            factor = 2**mip_level
            factor_tuple = (factor, factor, factor, 1)
            ds_start = [int(np.round(s / (2**mip_level))) for s in start]
            if not allow_non_aligned_write:
                # Align to chunk boundaries
                ds_start = [
                    int(round(math.floor(s / c) * c))
                    for s, c in zip(ds_start, chunk_size)
                ]
            downsample_shape = [
                int(math.ceil(s / f)) for s, f in zip(rawdata.shape, factor_tuple)
            ]
            ds_end_est = [s + d for s, d in zip(ds_start, downsample_shape)]
            if allow_non_aligned_write:
                bounds_from_end = [int(math.ceil(e / (2**mip_level))) for e in end]
                ds_end = [max(e1, e2) for e1, e2 in zip(ds_end_est, bounds_from_end)]
            else:
                # Align to chunk boundaries
                ds_end = [
                    int(round(math.ceil(e / c) * c))
                    for e, c in zip(ds_end_est, chunk_size)
                ]
                ds_end = [min(e, s) for e, s in zip(ds_end, volume_shape)]
            print("DS fill", ds_start, ds_end)
            downsampled = downsample_with_averaging(rawdata, factor_tuple)
            print("Downsampled shape:", downsampled.shape)

        if not allow_non_aligned_write:
            # TODO may need to ignore padding at the data edges
            # We may need to pad the downsampled data to fit the chunk boundaries
            pad_width = [
                (0, max(0, de - ds - s))
                for ds, de, s in zip(ds_start, ds_end, downsampled.shape)
            ]
            pad_width.append((0, 0))  # No padding for channel dimension
            # we should never pad more than the mip level times a factor inverse
            max_allowed_pad = 2 ** (num_mips - mip_level)
            max_actual_pad = max(pw[1] for pw in pad_width)
            if max_actual_pad > max_allowed_pad:
                raise ValueError(
                    f"Padding too large at mip {mip_level}: {pad_width}, max allowed {max_allowed_pad}"
                )
            if any(pw[1] > 0 for pw in pad_width):
                print("Padding downsampled data with:", pad_width)
                downsampled = np.pad(
                    downsampled, pad_width, mode="constant", constant_values=0
                )
                print("Padded downsampled shape:", downsampled.shape)

        vols[mip_level][
            ds_start[0] : ds_end[0], ds_start[1] : ds_end[1], ds_start[2] : ds_end[2]
        ] = downsampled

    # Mark chunk as complete
    touch(f_name)

    # Clean up cached file to save disk space
    delete_cached_zarr_file(x_i, y_i)

    # Return the bounds of the processed chunk
    return (start, end)


# %% Try with a single chunk to see if it works
x_i, y_i, z_i = 0, 0, 0
process((x_i, y_i, z_i))


# %% Loop over all the chunks
# Can do it in reverse order because the last chunks are most likely to error
in_reverse = False
coords = itertools.product(
    range(num_chunks_per_dim[0]),
    range(num_chunks_per_dim[1]),
    range(num_chunks_per_dim[2]),
)
if in_reverse:
    iter_coords = list(coords)
    iter_coords.reverse()
else:
    iter_coords = coords

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

# %% Function to check if a chunk is fully covered by processed bounds


def is_chunk_fully_covered(chunk_bounds, processed_chunks_bounds):
    """
    Check if a chunk is fully covered by processed bounds.

    Args:
        chunk_bounds: [start_coord, end_coord] where each coord is [x, y, z]
        processed_chunks_bounds: List of tuples (start, end) where start and end are [x, y, z]

    Returns:
        bool: True if all 8 corners of the chunk are covered by processed bounds
    """
    if not processed_chunks_bounds:
        return False

    start_coord, end_coord = chunk_bounds
    x0, y0, z0 = start_coord
    x1, y1, z1 = end_coord

    # Generate all 8 corners of the chunk
    corners = [
        [x0, y0, z0],  # min corner
        [x1, y0, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y1, z0],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],  # max corner
    ]

    # Check if each corner is covered by at least one processed bound
    for corner in corners:
        corner_covered = False
        for start, end in processed_chunks_bounds:
            # Check if corner is inside this processed bound
            if (
                start[0] <= corner[0] < end[0]
                and start[1] <= corner[1] < end[1]
                and start[2] <= corner[2] < end[2]
            ):
                corner_covered = True
                break

        # If any corner is not covered, the chunk is not fully covered
        if not corner_covered:
            return False

    # All corners are covered
    return True


# %% Read in the files that were already processed

already_uploaded_path = output_path / "uploaded_to_gcs_chunks.txt"
if already_uploaded_path.exists():
    with open(already_uploaded_path, "r") as f:
        uploaded_files = [line.strip() for line in f.readlines() if line.strip()]
else:
    uploaded_files = []

# %% Function to check the output directory for completed chunks and upload them to GCS

processed_chunks_bounds = []
failed_files = []


def upload_many_blobs_with_transfer_manager(
    bucket_name, filenames, source_directory="", workers=8
):
    """Upload every file in a list to a bucket, concurrently in a process pool.

    Each blob name is derived from the filename, not including the
    `source_directory` parameter. For complete control of the blob name for each
    file (and other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # A list (or other iterable) of filenames to upload.
    # filenames = ["file_1.txt", "file_2.txt"]

    # The directory on your computer that is the root of all of the files in the
    # list of filenames. This string is prepended (with os.path.join()) to each
    # filename to get the full path to the file. Relative paths and absolute
    # paths are both accepted. This string is not included in the name of the
    # uploaded blob; it is only used to find the source files. An empty string
    # means "the current working directory". Note that this parameter allows
    # directory traversal (e.g. "/", "../") and is not intended for unsanitized
    # end user input.
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client(project=gcs_project)
    bucket = storage_client.bucket(bucket_name)

    results = transfer_manager.upload_many_from_filenames(
        bucket,
        filenames,
        source_directory=source_directory,
        blob_name_prefix=gcs_output_path,
        max_workers=workers,
    )

    for name, result in zip(filenames, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            failed_files.append(name)
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            uploaded_files.append(name)


# TODO this probably wants to bulk together uploads to reduce overhead
def check_and_upload_completed_chunks():
    """
    Check for completed chunk files and upload them to GCS if configured.
    This helps manage local disk space by uploading and optionally removing completed chunks.

    Returns:
        int: Number of chunks uploaded
    """
    uploaded_count = 0
    files_to_upload_this_batch = []

    for mip_level in range(num_mips):
        factor = 2**mip_level
        dir_name = f"{factor}_{factor}_{factor}"
        output_path_for_mip = output_path / dir_name
        # For each file in the output dir check if it is fully covered by the already processed bounds
        # First, we loop over all the files in the output directory
        for chunk_file in output_path_for_mip.glob("**/*"):
            if str(chunk_file) in uploaded_files:
                continue
            # 1. Pull out the bounds of the chunk from the filename
            # filename format is x0-x1_y0-y1_z0-z1
            match = re.search(r"(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)", str(chunk_file))
            if not match:
                continue
            x0, x1, y0, y1, z0, z1 = map(int, match.groups())
            chunk_bounds = [(x0, y0, z0), (x1, y1, z1)]
            # Multiply by the factor to get back to original resolution
            chunk_bounds = [
                [c * factor for c in chunk_bounds[0]],
                [c * factor for c in chunk_bounds[1]],
            ]
            # Clamp the chunk bounds to the volume size
            chunk_bounds[1] = [
                min(cb, vs) for cb, vs in zip(chunk_bounds[1], volume_size)
            ]
            # Subtract 1 from the end bounds to make them inclusive
            chunk_bounds[1] = [cb - 1 for cb in chunk_bounds[1]]
            # 2. Check if the chunk is fully covered by the processed bounds
            covered = is_chunk_fully_covered(chunk_bounds, processed_chunks_bounds)

            if covered:
                # 3. If it is, mark to upload it to GCS
                files_to_upload_this_batch.append(chunk_file)

    if files_to_upload_this_batch:
        print(f"Uploading {len(files_to_upload_this_batch)} completed chunks to GCS...")
        if use_gcs_output:
            upload_many_blobs_with_transfer_manager(
                gcs_output_bucket_name, files_to_upload_this_batch, workers=8
            )
            uploaded_count += len(files_to_upload_this_batch)
        else:
            print("GCS output not configured, skipping upload")
            uploaded_count += len(files_to_upload_this_batch)
            uploaded_files.extend([str(x) for x in files_to_upload_this_batch])

        # Remove local chunks to save space
        if use_gcs_output and delete_output:
            for chunk_file in files_to_upload_this_batch:
                if chunk_file in failed_files:
                    print(f"Skipping deletion of failed upload chunk file {chunk_file}")
                    continue
                try:
                    chunk_file.unlink()
                except Exception as e:
                    print(f"Error deleting local chunk file {chunk_file}: {e}")

    # Append to the list of uploaded files
    with open(already_uploaded_path, "a") as f:
        for file in files_to_upload_this_batch:
            if file not in failed_files:
                f.write(f"{file}\n")

    return uploaded_count


def check_any_remaining_chunks():
    """
    Check any remaining chunks in the output directory to GCS.
    This is called at the end of processing to ensure all data is uploaded.

    """
    non_uploaded_files = []

    for mip_level in range(num_mips):
        factor = 2**mip_level
        dir_name = f"{factor}_{factor}_{factor}"
        output_path_for_mip = output_path / dir_name
        # For each file in the output dir
        for chunk_file in output_path_for_mip.glob("**/*"):
            if str(chunk_file) in uploaded_files:
                continue
            non_uploaded_files.append(str(chunk_file))

    return non_uploaded_files


# %% Move the data across with a single worker
total_uploaded_files = 0
# TEMP early quit for testing
max_iters = 4
for coord in iter_coords:
    bounds = process(coord)
    start, end = bounds
    processed_chunks_bounds.append((start, end))
    if max_iters and len(processed_chunks_bounds) >= max_iters:
        print("Reached max iterations for testing, stopping early")
        break

    # Periodically check and upload completed chunks to save disk space
    # This is done every 10 chunks to balance upload frequency vs overhead
    total_uploaded_files += check_and_upload_completed_chunks()
    print(f"Total uploaded chunks so far: {total_uploaded_files}")

# %% Show any failed uploads or files left over
if failed_files:
    print("The following files failed to upload to GCS:")
    for f in failed_files:
        print(f)

remaining_files = check_any_remaining_chunks()
if remaining_files:
    print(f"The following files were not uploaded yet: {remaining_files}")

# %% Serve the dataset to be used in neuroglancer
vols[0].viewer(port=1337)

# %%
