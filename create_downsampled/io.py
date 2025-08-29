# Mostly to handle local vs GCS paths

from pathlib import Path
import re

import zarr
import numpy as np

from chunking import is_chunk_fully_covered
from wells import extract_row_col_from_filename
from gcs import gcloud_download_dir, upload_many_blobs_with_transfer_manager


def get_local_cache_path(row, col, use_gcs_bucket, input_path):
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


def get_remote_file_path(row, col, total_rows, total_cols, all_files):
    """
    Get the remote file path (GCS blob name or local path) for a specific row and column.

    Args:
        row: Row index
        col: Column index

    Returns:
        str or Path: Remote file path, or None if not found
    """
    if row < 0 or row >= total_rows or col < 0 or col >= total_cols:
        raise ValueError(
            f"Row and column indices must be within the defined grid (0-{total_rows-1}, 0-{total_cols-1})."
        )

    # Try to find file by row/col pattern first
    for filepath in all_files:
        file_row, file_col = extract_row_col_from_filename(filepath)
        if file_row == row and file_col == col:
            return filepath

    return None


def download_zarr_file(
    row, col, use_gcs_bucket, input_path, total_rows, total_cols, all_files, gcs_project
):
    """
    Download the file for a specific row and column to local cache.

    Args:
        row: Row index
        col: Column index

    Returns:
        Path: Local cache path of downloaded file, or None if not found
    """
    remote_file = get_remote_file_path(row, col, total_rows, total_cols, all_files)
    if remote_file is None:
        print(f"No file found for row {row}, col {col}")
        return None

    local_path = get_local_cache_path(row, col, use_gcs_bucket, input_path)
    if local_path is None:
        return None

    # If file already exists locally, no need to download
    if local_path.exists():
        print(f"File already cached: {local_path}")
        return local_path

    local_path.mkdir(exist_ok=True, parents=True)

    if use_gcs_bucket:
        gcloud_download_dir(remote_file, local_path.parent, gcs_project)
        return local_path
    else:
        return remote_file  # For local files, just return the path


def load_file(
    row, col, use_gcs_bucket, input_path, total_rows, total_cols, all_files, gcs_project
):
    """
    Load the zarr store for a specific row and column.
    Downloads the file first if not already cached locally.

    Args:
        row: Row index
        col: Column index

    Returns:
        zarr store object, or None if not found/error
    """
    local_path = download_zarr_file(
        row,
        col,
        use_gcs_bucket,
        input_path,
        total_rows,
        total_cols,
        all_files,
        gcs_project,
    )
    if local_path is None:
        return None

    try:
        print(f"Loading zarr store from {local_path}")
        zarr_store = zarr.open(str(local_path), mode="r")
        return zarr_store
    except Exception as e:
        print(f"Error loading zarr store from {local_path}: {e}")
        return None


def delete_cached_zarr_file(row, col, use_gcs_bucket, delete_input, input_path):
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
    local_path = get_local_cache_path(row, col, use_gcs_bucket, input_path)
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


def load_zarr_store(file_path):
    """
    Load zarr store from a file path.
    This function is kept for backward compatibility.
    """
    zarr_store = zarr.open(str(file_path), mode="r")
    return zarr_store


def load_data_from_zarr_store(zarr_store, num_channels):
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


def get_uploaded_files(output_path):
    already_uploaded_path = output_path / "uploaded_to_gcs_chunks.txt"

    if already_uploaded_path.exists():
        with open(already_uploaded_path, "r") as f:
            uploaded_files = [line.strip() for line in f.readlines() if line.strip()]
    else:
        uploaded_files = []

    return uploaded_files


def check_and_upload_completed_chunks(
    num_mips,
    output_path,
    volume_size,
    processed_chunks_bounds,
    use_gcs_output,
    gcs_project,
    gcs_output_bucket_name,
    num_upload_workers,
    delete_output,
    already_uploaded_path,
    uploaded_files,
    failed_files,
):
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
            chunk_file = str(chunk_file)
            if chunk_file in uploaded_files:
                continue
            # 1. Pull out the bounds of the chunk from the filename
            # filename format is x0-x1_y0-y1_z0-z1
            match = re.search(r"(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)", chunk_file)
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
                gcs_output_bucket_name,
                files_to_upload_this_batch,
                source_directory=output_path,
                workers=num_upload_workers,
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


def check_any_remaining_chunks(num_mips, output_path, uploaded_files):
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
