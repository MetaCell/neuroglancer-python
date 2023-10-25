from typing import Any
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path
from math import ceil
from tqdm import tqdm
import json
import struct
from dataclasses import dataclass

import numpy as np
import dask.array as da

@dataclass
class Chunk:
    buffer: bytearray
    dimensions: list[int]
    
    def get_name(self):
        x_begin, x_end = self.dimensions[0], self.dimensions[1]
        y_begin, y_end = self.dimensions[2], self.dimensions[3]
        z_begin, z_end = self.dimensions[4], self.dimensions[5]
        return f"{x_begin}-{x_end}_{y_begin}-{y_end}_{z_begin}-{z_end}"
        

def load_data(input_filepath: Path) -> da.Array:
    """Load the OME-Zarr data and return a dask array"""
    url = parse_url(input_filepath)
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[0]
    return dask_data.persist()


def _create_metadata(
    chunk_size: tuple[int, int, int],
    block_size: tuple[int, int, int],
    data_size: tuple[int, int, int],
):
    metadata = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [list(chunk_size)],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": list(block_size),
                "resolution": [1, 1, 1],
                "key": "segmentation",
                "size": list(data_size),
            }
        ],
        "type": "segmentation",
    }
    return metadata


def write_segmentation(
    input_data: list[Chunk], metadata: dict[str, Any], output_directory: Path
):
    """Write the segmentation to the given directory"""
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_directory / "info"
    output_chunk_directory = output_directory / "segmentation"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    for chunk in input_data:
        output_filename = chunk.get_name()
        output_filepath = output_chunk_directory / output_filename
        with open(output_filepath, "wb") as f:
            f.write(input_data)


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


def _create_segmentation_chunk(dask_data, block_size=(8, 8, 8)):
    print(dask_data.shape)
    return 1
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

    # start_x, end_x = 
    return Chunk(buffer, [])

def create_segmentation(dask_data: da.Array, block_size):
    for block in dask_data.blocks:
        yield _create_segmentation_chunk(block, block_size)

def main(filename, block_size=(128, 128, 128)):
    dask_data = load_data(filename)
    metadata = _create_metadata(dask_data.chunksize, block_size, dask_data.shape)
    print(dask_data.chunks)
    chunks = [c for c in create_segmentation(dask_data, block_size)]
    write_segmentation(chunks, metadata, filename.parent / "segmentation")


if __name__ == "__main__":
    base_directory = Path("/media/starfish/LargeSSD/data/cryoET/data/segmentation")
    actin_filename = base_directory / "00004_actin_ground_truth_zarr"
    microtubules_filename = base_directory / "00004_MT_ground_truth_zarr"
    main(actin_filename)
    main(microtubules_filename)
