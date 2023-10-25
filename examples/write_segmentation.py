from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path
from math import ceil
from tqdm import tqdm
import struct

import numpy as np
import dask.array as da


def load_data(filename: Path) -> da.Array:
    """Load the OME-Zarr data and return a dask array"""
    url = parse_url(filename)
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[0]
    return dask_data


def write_segmentation():
    pass


def _get_grid_size_from_block_size(
    data_shape: tuple[int, int, int], block_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Calculate the grid size from the block size"""
    gx = ceil(data_shape[0] / block_size[0])
    gy = ceil(data_shape[1] / block_size[1])
    gz = ceil(data_shape[2] / block_size[2])
    return gx, gy, gz


def _get_buffer_position(buffer: bytearray) -> int:
    """Return the current position in the buffer"""
    if len(buffer) % 4 != 0:
        raise ValueError("Buffer length must be a multiple of 4")
    return len(buffer) // 4


def _pad_block(block: da.Array, block_size: tuple[int, int, int]) -> da.Array:
    """Pad the block to the given block size with zeros"""
    return da.pad(
        block,
        (
            (0, block_size[0] - block.shape[0]),
            (0, block_size[1] - block.shape[1]),
            (0, block_size[2] - block.shape[2]),
        ),
    )


def _create_block_header(
    buffer: bytearray,
    lookup_table_offset: int,
    encoded_bits: int,
    encoded_values_offset: int,
    block_offset: int,
):
    """
    Create a block header (64-bit)

    First 24 bits are the lookup table offset (little endian)
    Next 8 bits are the number of bits used to encode the values
    Last 32 bits are the offset to the encoded values (little endian)
    All values are unsigned integers

    Parameters
    ----------
    buffer : bytearray
        The buffer to write the block header to
    lookup_table_offset : int
        The offset in the buffer to the lookup table for this block
    encoded_bits : int
        The number of bits used to encode the values
    encoded_values_offset : int
        The offset in the buffer to the encoded values for this block
    block_offset : int
        The offset in the buffer to the block header
    """
    struct.pack_into(
        "<II",
        buffer,
        block_offset,
        lookup_table_offset | (encoded_bits << 24),
        encoded_values_offset,
    )


def _create_lookup_table(
    buffer: bytearray, stored_lookup_tables: dict[bytes, int], unique_values: da.Array
) -> tuple[int, int]:
    """
    Create a lookup table for the given values

    Parameters
    ----------
    buffer : bytearray
        The buffer to write the lookup table to
    stored_lookup_tables : dict[bytes, int]
        A dictionary mapping values to their offset in the buffer
    unique_values : np.ndarray
        The values to write to the buffer
        Must be uint32 or uint64

    Returns
    -------
    lookup_table_offset : int
        The offset in the buffer to the lookup table for the given values
    encoded_bits : int
        The number of bits used to encode the values
    """
    unique_values = unique_values.astype(np.uint32).compute()
    encoded_bits = int(np.ceil(np.log2(unique_values.shape[0])))
    if encoded_bits > 32:
        raise ValueError("Too many unique values in block")
    values_in_bytes = unique_values.tobytes()
    if values_in_bytes not in stored_lookup_tables:
        lookup_table_offset = _get_buffer_position(buffer)
        stored_lookup_tables[values_in_bytes] = lookup_table_offset
        buffer.extend(values_in_bytes)
    else:
        lookup_table_offset = stored_lookup_tables[values_in_bytes]
    return lookup_table_offset, encoded_bits


def _create_encoded_values(buffer: bytearray, positions: da.Array) -> int:
    """Create the encoded values for the given values

    Parameters
    ----------
    buffer: bytearray
        The buffer to write the encoded values to
    values: da.Array
        The values to encode

    Returns
    -------
    encoded_values_offset: int
        The offset in the buffer to the encoded values
    """
    encoded_values_offset = _get_buffer_position(buffer)
    buffer.extend(positions.astype(np.uint32).compute().tobytes())
    return encoded_values_offset


def convert_to_segmentation(dask_data, block_size=(8, 8, 8)):
    buffer = bytearray()
    bx, by, bz = block_size
    gx, gy, gz = _get_grid_size_from_block_size(dask_data.shape, block_size)
    stored_lookup_tables = {}

    for x, y, z in tqdm(np.ndindex(gx, gy, gz), total=gx * gy * gz):
        block = dask_data[
            x * bx : (x + 1) * bx, y * by : (y + 1) * by, z * bz : (z + 1) * bz
        ]
        unique_values, indices = da.unique(block, return_inverse=True)
        block = _pad_block(block, block_size)

        lookup_table_offset, encoded_bits = _create_lookup_table(
            buffer, stored_lookup_tables, unique_values
        )
        encoded_values_offset = _create_encoded_values(buffer, indices)
        block_offset = 8 * (x + gx * (y + gy * z))
        _create_block_header(
            buffer,
            lookup_table_offset,
            encoded_bits,
            encoded_values_offset,
            block_offset,
        )

    return buffer


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
