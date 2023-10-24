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


def _grid_size_from_block_size(data_shape, block_size):
    gx = ceil(data_shape[0] / block_size[0])
    gy = ceil(data_shape[1] / block_size[1])
    gz = ceil(data_shape[2] / block_size[2])
    return gx, gy, gz


def convert_to_segmentation(dask_data_arrays, block_size=(8, 8, 8)):
    shapes = [dask_data.shape for dask_data in dask_data_arrays]
    assert len(set(shapes)) == 1, "All data arrays must have the same shape"
    buffer = bytearray()
    bx, by, bz = block_size
    gx, gy, gz = _grid_size_from_block_size(dask_data_arrays[0].shape, block_size)
    stored_lookup_tables = {}

    temp_uniques = []

    for x, y, z in tqdm(np.ndindex(gx, gy, gz), total=gx * gy * gz):
        # Grab the data for this block
        temp = []
        for dask_data in dask_data_arrays:
            block = dask_data[
                x * bx : (x + 1) * bx, y * by : (y + 1) * by, z * bz : (z + 1) * bz
            ]
            #TODO pad the block to the block size
            unique_values, indices = da.unique(block, return_inverse=True)
            print(unique_values.compute())
            temp.append(indices)
        summed = da.sum(da.stack(temp, axis=0), axis=0)
        if da.any(summed > 1):
            logging.warning("Overlapping labels")
        
            temp_uniques.append(val)
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
    convert_to_segmentation(dask_data, (128, 128, 128))


if __name__ == "__main__":
    base_directory = Path("/media/starfish/LargeSSD/data/cryoET/data/segmentation")
    actin_filename = base_directory / "00004_actin_ground_truth_zarr"
    microtubules_filename = base_directory / "00004_MT_ground_truth_zarr"
    main((actin_filename, microtubules_filename))
