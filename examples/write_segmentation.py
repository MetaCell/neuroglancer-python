from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path
from math import ceil
from tqdm import tqdm
import logging

import numpy as np
import dask.array as da


def load_data(filename):
    url = parse_url(filename)
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[0]
    return dask_data


def write_segmentation():
    pass


def _get_grid_size_from_block_size(data_shape, block_size):
    gx = ceil(data_shape[0] / block_size[0])
    gy = ceil(data_shape[1] / block_size[1])
    gz = ceil(data_shape[2] / block_size[2])
    return gx, gy, gz


def _get_buffer_position(buffer):
    if len(buffer) % 4 != 0:
        raise ValueError("Buffer length must be a multiple of 4")
    return len(buffer) // 4


def convert_to_segmentation(dask_data, block_size=(8, 8, 8)):
    buffer = bytearray()
    bx, by, bz = block_size
    gx, gy, gz = _get_grid_size_from_block_size(dask_data.shape, block_size)
    stored_lookup_tables = {}

    temp_uniques = []

    for x, y, z in tqdm(np.ndindex(gx, gy, gz), total=gx * gy * gz):
        # Grab the data for this block
        block = dask_data[
            x * bx : (x + 1) * bx, y * by : (y + 1) * by, z * bz : (z + 1) * bz
        ]
        # TODO pad the block to the block size
        unique_values, indices = da.unique(block, return_inverse=True)
        unique_values = unique_values.astype(np.uint32).compute()
        nbits = int(np.ceil(np.log2(unique_values.shape[0])))
        if nbits > 32:
            raise ValueError("Too many unique values in block")
        values_in_bytes = unique_values.tobytes()
        if values_in_bytes not in stored_lookup_tables:
            offset = _get_buffer_position(buffer)
            stored_lookup_tables[values_in_bytes] = offset
            buffer.extend(values_in_bytes)
        else:
            offset = stored_lookup_tables[values_in_bytes]

        encoded_values_offset = _get_buffer_position(buffer)
        # TODO implement encoding
        buffer.extend(indices.astype(np.uint32).tobytes())

        # Create the block header
        # TODO improve but this is the basic idea
        block_header = np.array([nbits, offset, encoded_values_offset], dtype=np.uint32)

        # struct.pack_into("<II", buf, 8 * (x + gx * (y + gy * z)),
        #  lookup_table_offset | (bits << 24),
        #  encoded_values_offset)

        # unique_values_bytes = unique_values.astype(np.uint32).tobytes()
        # 1. block headers

        # 2. block data as (ordered by x, y, z)
        #   1. encoded values for the block
        #   2. lookup table for the block

    print(set(temp_uniques))
    # 2. block data as (ordered by x, y, z)
    #   1. encoded values for the block
    #   2. lookup table for the block


def main(filenames):
    dask_data_actin = load_data(filenames[0])
    dask_data_microtubules = load_data(filenames[1])
    dask_data = [dask_data_actin, dask_data_microtubules]
    convert_to_segmentation(dask_data[0], (128, 128, 128))


if __name__ == "__main__":
    base_directory = Path("/media/starfish/LargeSSD/data/cryoET/data/segmentation")
    actin_filename = base_directory / "00004_actin_ground_truth_zarr"
    microtubules_filename = base_directory / "00004_MT_ground_truth_zarr"
    main((actin_filename, microtubules_filename))
