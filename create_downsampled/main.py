# Call the relevant functions
import numpy as np

from load_config import load_env_config
from gcs import list_gcs_files, sync_info_to_gcs_output
from wells import compute_grid_dimensions, get_grid_coords
from gcs_local_io import (
    get_uploaded_files,
    load_file,
    check_and_upload_completed_chunks,
    check_any_remaining_chunks,
)
from chunking import compute_volume_and_chunk_size, process
from volume import create_cloudvolume_info

MAX_ITERS = 4  # For easier testing


def main():
    # The config is newer so use old var names just for ease
    config = load_env_config()
    use_gcs_bucket = config["USE_GCS_BUCKET"]

    gcs_bucket_name = config["GCS_BUCKET_NAME"]
    gcs_input_path = config["GCS_PREFIX"]
    gcs_file_ext = config["GCS_FILE_EXTENSION"]
    gcs_local_list = config["GCS_FILES_LOCAL_LIST"]
    gcs_project = config["GCS_PROJECT"]
    use_gcs_output = config["USE_GCS_OUTPUT"]
    gcs_output_bucket_name = config["GCS_OUTPUT_BUCKET_NAME"]
    gcs_output_path = config["GCS_OUTPUT_PREFIX"].rstrip("/") + "/"
    num_upload_workers = config["NUM_UPLOAD_WORKERS"]
    input_path = config["INPUT_PATH"]
    output_path = config["OUTPUT_PATH"]
    delete_input = config["DELETE_INPUT"]
    delete_output = config["DELETE_OUTPUT"]
    overwrite_output = config["OVERWRITE"]
    num_mips = config["NUM_MIPS"]
    mip_cutoff = config["MIP_CUTOFF"]
    channel_limit = config["CHANNEL_LIMIT"]
    allow_non_aligned_write = config["ALLOW_NON_ALIGNED_WRITE"]
    manual_chunk_size = config["MANUAL_CHUNK_SIZE"]

    # Now call the relevant functions to do the work
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the list of files to process
    if use_gcs_bucket:
        all_files = list_gcs_files(
            gcs_bucket_name,
            prefix=gcs_input_path,
            file_extension=gcs_file_ext,
            gcs_project=gcs_project,
            gcs_local_list=gcs_local_list,
        )
    else:
        all_files = list(input_path.glob(f"*{gcs_file_ext}"))
    all_files = sorted(all_files)

    # Get the well layout from the list of all files
    total_rows, total_cols = compute_grid_dimensions(all_files)

    # Determine sizes from the first file (assume all wells same)
    zarr_store = load_file(
        0, 0, use_gcs_bucket, input_path, total_rows, total_cols, all_files, gcs_project
    )
    single_vol_shape = zarr_store.shape
    # Input is in Z, T, C, Y, X order but want XYTCZ order
    single_file_xyz_shape = [
        single_vol_shape[4],
        single_vol_shape[3],
        single_vol_shape[1],
    ]
    single_file_shape = np.array(single_file_xyz_shape)

    # Compute volume and chunk sizes
    volume_size, chunk_size = compute_volume_and_chunk_size(
        single_file_xyz_shape,
        total_rows,
        total_cols,
        num_mips,
        manual_chunk_size,
    )

    num_chunks_per_dim = np.ceil(volume_size / single_file_shape).astype(int)
    num_channels = min(single_vol_shape[2], channel_limit)
    data_type = "uint16"

    vols = create_cloudvolume_info(
        num_channels,
        data_type,
        num_mips,
        volume_size,
        chunk_size,
        output_path,
        allow_non_aligned_write=allow_non_aligned_write,
    )

    # Process each well into chunks
    iter_coords = list(get_grid_coords(num_chunks_per_dim))

    # Find which files were already done and keep track of them
    uploaded_files = get_uploaded_files(output_path)

    processed_chunks = []
    failed_chunks = []
    total_uploads = 0
    for coord in iter_coords[:MAX_ITERS]:
        bounds = process(
            args=coord,
            single_file_shape=single_file_shape,
            volume_shape=volume_size,
            vols=vols,
            chunk_size=chunk_size,
            num_mips=num_mips,
            mip_cutoff=mip_cutoff,
            allow_non_aligned_write=allow_non_aligned_write,
            overwrite_output=overwrite_output,
            progress_dir=output_path / "progress",
            total_rows=total_rows,
            total_cols=total_cols,
            use_gcs_bucket=use_gcs_bucket,
            input_path=input_path,
            all_files=all_files,
            delete_input=delete_input,
            gcs_project=gcs_project,
            num_channels=num_channels,
        )
        start, end = bounds
        processed_chunks.append((start, end))
        total_uploads += check_and_upload_completed_chunks(
            num_mips=num_mips,
            output_path=output_path,
            volume_size=volume_size,
            processed_chunks_bounds=processed_chunks,
            use_gcs_output=use_gcs_output,
            gcs_project=gcs_project,
            gcs_output_bucket_name=gcs_output_bucket_name,
            gcs_output_path=gcs_output_path,
            num_upload_workers=num_upload_workers,
            delete_output=delete_output,
            already_uploaded_path=output_path / "uploaded_to_gcs_chunks.txt",
            uploaded_files=uploaded_files,
            failed_files=failed_chunks,
        )
        print(f"Total chunks uploaded so far: {total_uploads}")

    if failed_chunks:
        print("Some chunks failed to upload, writing to failed_chunks.txt")
        with open(output_path / "failed_chunks.txt", "w") as f:
            for item in failed_chunks:
                f.write(f"{item}\n")

    remaining_files = check_any_remaining_chunks(
        num_mips=num_mips, output_path=output_path, uploaded_files=uploaded_files
    )
    if remaining_files:
        for f in remaining_files:
            if f not in failed_chunks:
                print(f"Remaining file not yet uploaded: {f}")
    else:
        # Do at the end to avoid uploading info file repeatedly
        # if issues during processing of chunks
        sync_info_to_gcs_output(
            output_path,
            gcs_output_path,
            use_gcs_output,
            gcs_project,
            gcs_output_bucket_name,
        )


if __name__ == "__main__":
    main()
