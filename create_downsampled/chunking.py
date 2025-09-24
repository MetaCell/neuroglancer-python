import math

from cloudvolume.lib import touch
import numpy as np
from neuroglancer.downsample import downsample_with_averaging

from gcs_local_io import load_data_from_zarr_store, delete_cached_zarr_file, load_file


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


def compute_volume_and_chunk_size(
    single_file_xyz_shape,
    computed_num_rows,
    computed_num_cols,
    num_mips,
    manual_chunk_size,
):
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
        computed_chunk_size = compute_optimal_chunk_size(
            single_file_xyz_shape, num_mips
        )
        print(f"Computed optimal chunk size: {computed_chunk_size}")
        chunk_size = computed_chunk_size

    volume_size = [
        single_file_xyz_shape[0] * computed_num_cols,
        single_file_xyz_shape[1] * computed_num_rows,
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

    return volume_size, chunk_size


def process(
    args,
    single_file_shape,
    volume_shape,
    vols,
    chunk_size,
    num_mips,
    mip_cutoff,
    allow_non_aligned_write,
    overwrite_output,
    progress_dir,
    total_rows,
    total_cols,
    use_gcs_bucket,
    input_path,
    all_files,
    delete_input,
    gcs_project,
    num_channels,
):
    y_i, x_i, z_i = args
    x_file = args[0]
    y_file = args[1]

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
        return (start, end), False

    print(
        f"Loading file for coordinates at ({x_i}, {y_i}, {z_i}), file index r{x_file}, c{y_file}"
    )
    loaded_zarr_store = load_file(
        x_file,
        y_file,
        use_gcs_bucket,
        input_path,
        total_rows,
        total_cols,
        all_files,
        gcs_project,
    )

    if loaded_zarr_store is None:
        print(
            f"Warning: Could not load file for row {x_file}, col {y_file}. Skipping..."
        )
        return

    rawdata = load_data_from_zarr_store(loaded_zarr_store, num_channels)

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
            downsampled = downsample_with_averaging(rawdata, factor_tuple)

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

        # We need to ensure that the end does not exceed the volume size at this mip level
        mip_bounds = [
            max(1, round(volume_shape[0] / (2**mip_level))),
            max(1, round(volume_shape[1] / (2**mip_level))),
            max(1, round(volume_shape[2] / (2**mip_level))),
        ]
        ds_end = [min(e, mb) for e, mb in zip(ds_end, mip_bounds)]
        downsample_size = [e - s for s, e in zip(ds_start, ds_end)]
        print(
            f"MIP {mip_level} (bounds {mip_bounds}): Writing data to volume at {ds_start} to {ds_end}"
        )

        sliced_ds = downsampled[
            : downsample_size[0], : downsample_size[1], : downsample_size[2], :
        ]
        # In case the requested bounds are a little larger than the downsampled data due to rounding
        mip_ends = [s + d for s, d in zip(ds_start, sliced_ds.shape)]
        vols[mip_level][
            ds_start[0] : mip_ends[0],
            ds_start[1] : mip_ends[1],
            ds_start[2] : mip_ends[2],
            :,
        ] = sliced_ds

    # Mark chunk as complete
    touch(f_name)

    # Clean up cached file to save disk space
    # Don't delete row 0 col 0 though as that's a reference file
    if not (x_i == 0 and y_i == 0):
        delete_cached_zarr_file(
            x_file,
            y_file,
            total_rows,
            total_cols,
            use_gcs_bucket,
            delete_input,
            input_path,
            all_files,
        )

    # Return the bounds of the processed chunk
    return (start, end), True
