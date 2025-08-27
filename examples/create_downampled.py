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
    print(
        "Warning: python-dotenv not installed. Install with 'pip install python-dotenv' for .env file support."
    )

import re
import tempfile
import shutil

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
        
        # Output GCS bucket configuration (for uploading results)
        "USE_GCS_OUTPUT": parse_bool(os.getenv("USE_GCS_OUTPUT", "false")),
        "GCS_OUTPUT_BUCKET_NAME": os.getenv("GCS_OUTPUT_BUCKET_NAME", "your-output-bucket-name"),
        "GCS_OUTPUT_PREFIX": os.getenv("GCS_OUTPUT_PREFIX", "processed/"),
        
        # Local paths (used when USE_GCS_BUCKET is False)
        "INPUTFOLDER": Path(os.getenv("INPUTFOLDER", "/temp/in")),
        "OUTPUT_PATH": Path(os.getenv("OUTPUT_PATH", "/temp/out")),
        # Processing settings
        "OVERWRITE": parse_bool(os.getenv("OVERWRITE", "false")),
        "NUM_MIPS": int(os.getenv("NUM_MIPS", "5")),
        "MIP_CUTOFF": int(os.getenv("MIP_CUTOFF", "0")),
        "CHANNEL_LIMIT": int(os.getenv("CHANNEL_LIMIT", "4")),
        "NUM_ROWS": int(os.getenv("NUM_ROWS", "3")),
        "NUM_COLS": int(os.getenv("NUM_COLS", "6")),
        "ALLOW_NON_ALIGNED_WRITE": parse_bool(
            os.getenv("ALLOW_NON_ALIGNED_WRITE", "false")
        ),
        # Optional resolution settings
        "SIZE_X": int(os.getenv("SIZE_X", "1")),
        "SIZE_Y": int(os.getenv("SIZE_Y", "1")),
        "SIZE_Z": int(os.getenv("SIZE_Z", "1")),
        # Optional chunk settings
        "CHUNK_SIZE_X": int(os.getenv("CHUNK_SIZE_X", "64")),
        "CHUNK_SIZE_Y": int(os.getenv("CHUNK_SIZE_Y", "64")),
        "CHUNK_SIZE_Z": int(os.getenv("CHUNK_SIZE_Z", "32")),
    }

    return config


# Load configuration
config = load_env_config()

# Extract configuration variables for backward compatibility
USE_GCS_BUCKET = config["USE_GCS_BUCKET"]
GCS_BUCKET_NAME = config["GCS_BUCKET_NAME"]
GCS_PREFIX = config["GCS_PREFIX"]
GCS_FILE_EXTENSION = config["GCS_FILE_EXTENSION"]
USE_GCS_OUTPUT = config["USE_GCS_OUTPUT"]
GCS_OUTPUT_BUCKET_NAME = config["GCS_OUTPUT_BUCKET_NAME"]
GCS_OUTPUT_PREFIX = config["GCS_OUTPUT_PREFIX"]
INPUTFOLDER = config["INPUTFOLDER"]
OUTPUT_PATH = config["OUTPUT_PATH"]
OVERWRITE = config["OVERWRITE"]
NUM_MIPS = config["NUM_MIPS"]
MIP_CUTOFF = config["MIP_CUTOFF"]
CHANNEL_LIMIT = config["CHANNEL_LIMIT"]
NUM_ROWS = config["NUM_ROWS"]
NUM_COLS = config["NUM_COLS"]
ALLOW_NON_ALIGNED_WRITE = config["ALLOW_NON_ALIGNED_WRITE"]

# Print loaded configuration for verification
print("Configuration loaded:")
print(f"  Data source: {'GCS Bucket' if USE_GCS_BUCKET else 'Local files'}")
if USE_GCS_BUCKET:
    print(f"  GCS Bucket: {GCS_BUCKET_NAME}")
    print(f"  GCS Prefix: {GCS_PREFIX}")
else:
    print(f"  Input folder: {INPUTFOLDER}")
print(f"  Output path: {OUTPUT_PATH}")
print(f"  Output to GCS: {'Yes' if USE_GCS_OUTPUT else 'No'}")
if USE_GCS_OUTPUT:
    print(f"  Output GCS Bucket: {GCS_OUTPUT_BUCKET_NAME}")
    print(f"  Output GCS Prefix: {GCS_OUTPUT_PREFIX}")
print(
    f"  Processing: {NUM_MIPS} mips, {CHANNEL_LIMIT} channels, {NUM_ROWS}x{NUM_COLS} grid"
)

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
    if USE_GCS_BUCKET:
        files = list_gcs_files(GCS_BUCKET_NAME, GCS_PREFIX, GCS_FILE_EXTENSION)
    else:
        # Use local filesystem glob as before
        files = list(INPUTFOLDER.glob(f"**/*{GCS_FILE_EXTENSION}"))
    print(f"Total files found: {len(files)}")
    return sorted(files)


# Get the list of available files
all_files = get_file_list()


def extract_row_col_from_filename(filename):
    """
    Extract row and column information from filename.
    Assumes filenames contain row/col info in a pattern like 'r{row}c{col}' or '{row}_{col}'.
    This function should be customized based on your actual filename pattern.

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

    # Try pattern where position in sorted list determines row/col
    # This is a fallback - assumes files are sorted in row-major order
    return None, None


def compute_grid_dimensions(file_list):
    """
    Compute the number of rows and columns from the available files.

    Args:
        file_list: List of file paths or names

    Returns:
        tuple: (num_rows, num_cols)
    """
    if not file_list:
        return 0, 0

    rows_found = set()
    cols_found = set()

    for filepath in file_list:
        row, col = extract_row_col_from_filename(filepath)
        if row is not None and col is not None:
            rows_found.add(row)
            cols_found.add(col)

    if rows_found and cols_found:
        num_rows = max(rows_found) + 1  # Assuming 0-indexed
        num_cols = max(cols_found) + 1  # Assuming 0-indexed
        print(
            f"Detected grid dimensions from filenames: {num_rows} rows × {num_cols} columns"
        )
        return num_rows, num_cols
    else:
        # Fallback: try to infer from total number of files
        total_files = len(file_list)
        # Try to find a reasonable rectangular arrangement
        import math

        num_cols = int(math.sqrt(total_files))
        num_rows = math.ceil(total_files / num_cols)
        print(
            f"Could not detect grid dimensions from filenames. Using fallback: {num_rows} rows × {num_cols} columns for {total_files} files"
        )
        return num_rows, num_cols


# Compute actual grid dimensions from files
COMPUTED_NUM_ROWS, COMPUTED_NUM_COLS = compute_grid_dimensions(all_files)

# Use computed dimensions if they seem reasonable, otherwise fall back to config
if COMPUTED_NUM_ROWS > 0 and COMPUTED_NUM_COLS > 0:
    NUM_ROWS = COMPUTED_NUM_ROWS
    NUM_COLS = COMPUTED_NUM_COLS
    print(f"Using computed grid dimensions: {NUM_ROWS} rows × {NUM_COLS} columns")
else:
    print(f"Using configured grid dimensions: {NUM_ROWS} rows × {NUM_COLS} columns")

# %% File management functions
#
# The new file management system provides three main functions:
#
# 1. download_file(row, col): Downloads file from GCS/copies from local to cache
# 2. load_file(row, col): Downloads (if needed) and loads zarr store
# 3. delete_cached_file(row, col): Removes cached file to save disk space
#
# The system automatically handles:
# - Downloading from GCS bucket or copying from local filesystem
# - Caching files locally to avoid repeated downloads
# - Row/column to filename mapping (tries pattern matching first, falls back to index)
# - Error handling and logging
#
# Usage examples:
# - zarr_store = load_file(0, 1)  # Load file for row 0, column 1
# - delete_cached_file(0, 1)      # Delete cached file to free space

# %% GCS Output Functions
#
# The GCS output system provides functionality to upload processed results to a Google Cloud bucket:
#
# 1. sync_info_to_gcs_output(): Uploads CloudVolume info/provenance files after creation
# 2. upload_chunk_to_gcs(): Uploads individual chunk files to GCS
# 3. check_and_upload_completed_chunks(): Batch upload of completed chunks
#
# Configuration (set in .env file):
# - USE_GCS_OUTPUT=true: Enable GCS output functionality  
# - GCS_OUTPUT_BUCKET_NAME: Name of destination GCS bucket
# - GCS_OUTPUT_PREFIX: Path prefix within bucket (e.g. "processed/dataset1/")
#
# The system automatically:
# - Uploads info files immediately after CloudVolume creation
# - Periodically uploads chunks during processing to manage disk space
# - Performs final upload of any remaining chunks at completion
#
# Benefits:
# - Manages local disk space by uploading completed chunks
# - Provides distributed access to processed datasets
# - Enables incremental processing across multiple machines


def get_local_cache_path(row, col):
    """
    Get the local cache path where a file for the given row/col should be stored.

    Args:
        row: Row index
        col: Column index

    Returns:
        Path: Local cache path for the file
    """
    cache_dir = OUTPUT_PATH / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Get the remote file path/name for this row/col
    remote_file = get_remote_file_path(row, col)
    if remote_file is None:
        return None

    # Create local filename based on remote file
    if USE_GCS_BUCKET:
        # For GCS files, use the blob name but replace slashes with underscores
        local_name = str(remote_file).replace("/", "_").replace("\\", "_")
    else:
        # For local files, just use the filename
        local_name = Path(remote_file).name

    return cache_dir / local_name


def get_remote_file_path(row, col):
    """
    Get the remote file path (GCS blob name or local path) for a specific row and column.

    Args:
        row: Row index
        col: Column index

    Returns:
        str or Path: Remote file path, or None if not found
    """
    if row < 0 or row >= NUM_ROWS or col < 0 or col >= NUM_COLS:
        raise ValueError(
            f"Row and column indices must be within the defined grid (0-{NUM_ROWS-1}, 0-{NUM_COLS-1})."
        )

    # Try to find file by row/col pattern first
    for filepath in all_files:
        file_row, file_col = extract_row_col_from_filename(filepath)
        if file_row == row and file_col == col:
            return filepath

    # Fallback: use index-based access (assumes row-major order)
    index = row * NUM_COLS + col
    if index < len(all_files):
        return all_files[index]

    return None


def download_file(row, col):
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

    if USE_GCS_BUCKET:
        # Download from GCS
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(remote_file)

            print(f"Downloading {remote_file} to {local_path}")
            blob.download_to_filename(str(local_path))
            print(f"Downloaded successfully: {local_path}")
            return local_path
        except Exception as e:
            print(f"Error downloading {remote_file}: {e}")
            return None
    else:
        # Copy from local filesystem
        try:
            print(f"Copying {remote_file} to {local_path}")
            shutil.copy2(remote_file, local_path)
            print(f"Copied successfully: {local_path}")
            return local_path
        except Exception as e:
            print(f"Error copying {remote_file}: {e}")
            return None


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
    local_path = download_file(row, col)
    if local_path is None:
        return None

    try:
        print(f"Loading zarr store from {local_path}")
        zarr_store = zarr.open(str(local_path), mode="r")
        return zarr_store
    except Exception as e:
        print(f"Error loading zarr store from {local_path}: {e}")
        return None


def delete_cached_file(row, col):
    """
    Delete the locally cached file for a specific row and column to save disk space.

    Args:
        row: Row index
        col: Column index

    Returns:
        bool: True if file was deleted or didn't exist, False if error
    """
    local_path = get_local_cache_path(row, col)
    if local_path is None:
        return True

    try:
        if local_path.exists():
            local_path.unlink()
            print(f"Deleted cached file: {local_path}")
        return True
    except Exception as e:
        print(f"Error deleting cached file {local_path}: {e}")
        return False


# Backward compatibility function that uses the new structure
def get_file_for_row_col(row, col):
    """
    Get the file path for a specific row and column.
    This is kept for backward compatibility, but the new approach is to use
    load_file() which handles download automatically.
    """
    return get_remote_file_path(row, col)


def sync_info_to_gcs_output():
    """
    Sync the CloudVolume info file to the GCS output bucket.
    This uploads the info file so the bucket is ready to receive the rest of the data.
    """
    if not USE_GCS_OUTPUT:
        print("GCS output not enabled, skipping info sync")
        return True
        
    print(f"Syncing info file to GCS output bucket: {GCS_OUTPUT_BUCKET_NAME}")
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_OUTPUT_BUCKET_NAME)
        
        # The local info file path
        local_info_path = OUTPUT_PATH / "info"
        
        if not local_info_path.exists():
            print(f"Warning: Local info file not found at {local_info_path}")
            return False
        
        # The GCS destination path for the info file
        gcs_info_path = GCS_OUTPUT_PREFIX.rstrip('/') + '/info'
        
        # Upload the info file
        blob = bucket.blob(gcs_info_path)
        blob.upload_from_filename(str(local_info_path))
        print(f"Uploaded info file to gs://{GCS_OUTPUT_BUCKET_NAME}/{gcs_info_path}")
        
        # Also upload provenance if it exists
        local_provenance_path = OUTPUT_PATH / "provenance"
        if local_provenance_path.exists():
            gcs_provenance_path = GCS_OUTPUT_PREFIX.rstrip('/') + '/provenance'
            provenance_blob = bucket.blob(gcs_provenance_path)
            provenance_blob.upload_from_filename(str(local_provenance_path))
            print(f"Uploaded provenance file to gs://{GCS_OUTPUT_BUCKET_NAME}/{gcs_provenance_path}")
        
        return True
        
    except Exception as e:
        print(f"Error syncing info to GCS output bucket: {e}")
        return False

def upload_chunk_to_gcs(local_chunk_path, gcs_chunk_path):
    """
    Upload a single chunk file to the GCS output bucket.
    
    Args:
        local_chunk_path: Path to local chunk file
        gcs_chunk_path: GCS blob path for the chunk
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not USE_GCS_OUTPUT:
        return True
        
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_OUTPUT_BUCKET_NAME)
        
        blob = bucket.blob(gcs_chunk_path)
        blob.upload_from_filename(str(local_chunk_path))
        
        return True
        
    except Exception as e:
        print(f"Error uploading chunk {local_chunk_path} to GCS: {e}")
        return False


def check_and_upload_completed_chunks():
    """
    Check for completed chunk files and upload them to GCS if configured.
    This helps manage local disk space by uploading and optionally removing completed chunks.
    
    Returns:
        int: Number of chunks uploaded
    """
    if not USE_GCS_OUTPUT:
        return 0
        
    uploaded_count = 0
    
    try:
        # Look for chunk files in the output directory
        for mip_level in range(NUM_MIPS):
            mip_dir = OUTPUT_PATH / str(mip_level)
            if mip_dir.exists():
                # Find all chunk files in this mip level
                for chunk_file in mip_dir.glob('**/*'):
                    if chunk_file.is_file():
                        # Construct the GCS path for this chunk
                        relative_path = chunk_file.relative_to(OUTPUT_PATH)
                        gcs_chunk_path = GCS_OUTPUT_PREFIX.rstrip('/') + '/' + str(relative_path).replace('\\', '/')
                        
                        # Check if chunk should be uploaded (you can add more logic here)
                        if upload_chunk_to_gcs(chunk_file, gcs_chunk_path):
                            uploaded_count += 1
                            print(f"Uploaded chunk: {gcs_chunk_path}")
                            
                            # Optionally remove local chunk to save space
                            # Uncomment the next line if you want to delete local chunks after upload
                            # chunk_file.unlink()
        
        if uploaded_count > 0:
            print(f"Uploaded {uploaded_count} chunks to GCS output bucket")
            
    except Exception as e:
        print(f"Error checking/uploading chunks: {e}")
    
    return uploaded_count


def load_zarr_store(file_path):
    """
    Load zarr store from a file path.
    This function is kept for backward compatibility.
    """
    zarr_store = zarr.open(str(file_path), mode="r")
    return zarr_store


def load_data_from_zarr_store(zarr_store):
    # Input is in Z, T, C, Y, X order

    num_channels = min(zarr_store.shape[2], CHANNEL_LIMIT)
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


# Load the first file to inspect data shape and properties
print("Loading first file for data inspection...")
zarr_store = load_file(0, 0)  # Load file at row 0, col 0
if zarr_store is None:
    print("Error: Could not load first file for data inspection")
    exit(1)

# %% Inspect the data
shape = zarr_store.shape
# Input is in Z, T, C, Y, X order
# Want XYTCZ order
single_file_xyz_shape = [shape[4], shape[3], shape[1]]
size_x = config["SIZE_X"]
size_y = config["SIZE_Y"]
size_z = config["SIZE_Z"]
# Here, T and Z are kind of transferrable.
# The reason is because the z dimension in neuroglancer is the time dimension
# from the raw original data.
# So both terms might be used to represent the same thing.
# It's a bit unusual that t is being used as the z dimension,
# but otherwise you can't do volume rendering in neuroglancer.

num_channels = min(shape[2], CHANNEL_LIMIT)  # Limit to NUM_CHANNELS for memory usage
data_type = "uint16"

def compute_optimal_chunk_size(single_file_shape, num_mips, max_chunk_size=None):
    """
    Compute optimal chunk size based on single file shape and number of MIP levels.
    
    The goal is to choose chunk sizes that:
    1. Divide evenly into the data dimensions at all MIP levels
    2. Are powers of 2 for optimal octree alignment
    3. Are reasonable in terms of memory usage
    4. Work well with the downsampling structure
    
    Args:
        single_file_shape: [x, y, z] shape of a single file
        num_mips: Number of MIP levels
        max_chunk_size: Optional maximum chunk size (default: 512)
        
    Returns:
        List[int]: [chunk_x, chunk_y, chunk_z] optimal chunk sizes
    """
    if max_chunk_size is None:
        max_chunk_size = 512
    
    optimal_chunks = []
    
    for dim_size in single_file_shape:
        # Find the largest power of 2 that:
        # 1. Is <= max_chunk_size
        # 2. Divides reasonably into the dimension size
        # 3. Works well across all MIP levels
        
        # Start with powers of 2 up to max_chunk_size
        candidate_chunks = []
        power = 1
        while power <= max_chunk_size:
            candidate_chunks.append(power)
            power *= 2
        
        # Score each candidate based on how well it divides the data
        best_chunk = candidate_chunks[0]
        best_score = float('inf')
        
        for chunk_size in candidate_chunks:
            # Score based on how well it divides across MIP levels
            score = 0
            
            for mip in range(num_mips):
                # At each MIP level, data is downsampled by 2^mip
                effective_dim_size = dim_size // (2 ** mip)
                if effective_dim_size > 0:
                    # Prefer chunk sizes that divide evenly
                    remainder = effective_dim_size % chunk_size
                    score += remainder ** 2  # Penalize remainders quadratically
                    
                    # Also prefer chunk sizes that don't create too many tiny chunks
                    num_chunks = math.ceil(effective_dim_size / chunk_size)
                    if num_chunks > 0:
                        avg_chunk_fill = effective_dim_size / (num_chunks * chunk_size)
                        score += (1 - avg_chunk_fill) ** 2 * 100  # Penalize poor utilization
            
            # Additional penalty for very small chunks
            if chunk_size < 32:
                score += 1000
            
            if score < best_score:
                best_score = score
                best_chunk = chunk_size
        
        optimal_chunks.append(best_chunk)
    
    return optimal_chunks

# Compute optimal chunk size based on single file shape and MIP levels
print(f"Computing optimal chunk size for shape {single_file_xyz_shape} with {NUM_MIPS} MIP levels...")
computed_chunk_size = compute_optimal_chunk_size(single_file_xyz_shape, NUM_MIPS)

# Allow override from config, but show the computed recommendation
config_chunk_size = [
    config["CHUNK_SIZE_X"],
    config["CHUNK_SIZE_Y"],
    config["CHUNK_SIZE_Z"],
]

print(f"Computed optimal chunk size: {computed_chunk_size}")
print(f"Config chunk size: {config_chunk_size}")

# Use computed chunk size, but allow config override if set to non-default values
use_computed = True
default_chunk = [64, 64, 32]  # Default values from .env.example
if config_chunk_size != default_chunk:
    print("Using chunk size from configuration (non-default values detected)")
    chunk_size = config_chunk_size
    use_computed = False
else:
    print("Using computed optimal chunk size")
    chunk_size = computed_chunk_size

print(f"Final chunk size: {chunk_size}")

# Validate chunk size works with the data
for i, (dim_name, dim_size, chunk_dim) in enumerate(zip(['X', 'Y', 'Z'], single_file_xyz_shape, chunk_size)):
    num_chunks_this_dim = math.ceil(dim_size / chunk_dim)
    print(f"  {dim_name} dimension: {dim_size} → {num_chunks_this_dim} chunks of size {chunk_dim}")
    
    # Check how this works across MIP levels
    for mip in range(min(3, NUM_MIPS)):  # Show first few MIP levels
        effective_size = dim_size // (2 ** mip)
        if effective_size > 0:
            mip_chunks = math.ceil(effective_size / chunk_dim)
            utilization = (effective_size / (mip_chunks * chunk_dim)) * 100 if mip_chunks > 0 else 0
            print(f"    MIP {mip}: {effective_size} → {mip_chunks} chunks ({utilization:.1f}% utilization)")
# The chunk size remains fixed across all mips, but at higher mips
# the data will be downsampled, so the effective chunk size will be larger.
# Because we use precomputed data format, every chunk has to have all channels included.
# The optimal chunk size balances:
# - Memory usage (smaller chunks use less memory)
# - I/O efficiency (larger chunks reduce overhead)
# - Octree alignment (powers of 2 work best)
# - Downsampling efficiency (should divide well at all MIP levels)

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

# Sync the info file to GCS output bucket if configured
sync_info_to_gcs_output()

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

# Info file synced to GCS output bucket (if configured)
# The sync_info_to_gcs_output() function above handles uploading the info and provenance files
# to the configured GCS output bucket so it's ready to receive chunk data.

# %% Functions for moving data
shape = volume_size
chunk_shape = np.array([1500, 936, 687])  # this is for reading data
num_chunks_per_dim = np.ceil(shape / chunk_shape).astype(int)


def process(args):
    x_i, y_i, z_i = args

    # Use the new load_file function that handles download/caching
    print(f"Loading file for coordinates ({x_i}, {y_i}, {z_i})")
    loaded_zarr_store = load_file(x_i, y_i)

    if loaded_zarr_store is None:
        print(f"Warning: Could not load file for row {x_i}, col {y_i}. Skipping...")
        return

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

    # Process all mip levels
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

    # Mark chunk as complete
    touch(f_name)

    # Clean up cached file to save disk space
    # (you can comment this out if you want to keep files cached)
    delete_cached_file(x_i, y_i)


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
chunk_count = 0
for coord in reversed_coords:
    process(coord)
    chunk_count += 1
    
    # Periodically check and upload completed chunks to save disk space
    # This is done every 10 chunks to balance upload frequency vs overhead
    if USE_GCS_OUTPUT and chunk_count % 10 == 0:
        print(f"Processed {chunk_count} chunks, checking for uploads...")
        check_and_upload_completed_chunks()
    
    # The original TODO was about uploading chunks as they're completed
    # The above implementation provides a basic version of this functionality
    # For more sophisticated chunk management, you could:
    # 1. Track which specific chunks are complete across all MIP levels
    # 2. Only upload chunks that are fully written at all relevant MIP levels  
    # 3. Implement more granular deletion of local chunks after successful upload
    # 4. Add retry logic for failed uploads

# Final upload of any remaining chunks
if USE_GCS_OUTPUT:
    print("Processing complete, uploading any remaining chunks...")
    final_upload_count = check_and_upload_completed_chunks()
    print(f"Final upload completed: {final_upload_count} chunks uploaded")

# %% Serve the dataset to be used in neuroglancer
vols[0].viewer(port=1337)

# %%
