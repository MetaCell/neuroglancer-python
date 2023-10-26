from math import ceil
import struct

import numpy as np
import dask.array as da

from neuroglancer_data_conversion.utils import (
    pad_block,
    get_grid_size_from_block_size,
)
from neuroglancer_data_conversion.chunk import Chunk

DEBUG = False


def _get_buffer_position(buffer: bytearray) -> int:
    """Return the current position in the buffer"""
    if len(buffer) % 4 != 0:
        raise ValueError("Buffer length must be a multiple of 4")
    return len(buffer) // 4


def _get_encoded_bits(unique_values: np.ndarray) -> int:
    """Return the number of bits needed to encode the given values"""
    if np.all(unique_values == 0) or len(unique_values) == 0:
        return 0
    bits = 1
    while 2**bits < len(unique_values):
        bits += 1
    if bits > 32:
        raise ValueError("Too many unique values in block")
    return bits


def _pack_encoded_values(encoded_values: np.ndarray, encoded_bits: int) -> bytes:
    """
    Pack the encoded values into 32bit unsigned integers

    To view the packed values as a numpy array, use the following:
    np.frombuffer(packed_values, dtype=np.uint32).view(f"u{encoded_bits}")

    Parameters
    ----------
    encoded_values : np.ndarray
        The encoded values
    encoded_bits : int
        The number of bits used to encode the values

    Returns
    -------
    packed_values : bytes
        The packed values
    """
    if encoded_bits == 0:
        return bytes()
    values_per_uint32 = 32 // encoded_bits
    number_of_values = ceil(len(encoded_values) / values_per_uint32)
    padded_values = np.pad(
        encoded_values,
        (0, number_of_values * values_per_uint32 - len(encoded_values)),
    )
    if encoded_bits == 1:
        reshaped = padded_values.reshape((-1, 32)).astype(np.uint8)
        packed_values = np.packbits(reshaped, bitorder="little").tobytes()

        if DEBUG:
            values, binary = get_back_values_from_buffer(packed_values)
            assert np.all(binary == padded_values)
        return packed_values
    else:
        raise NotImplementedError("Only 1 bit encoding is implemented")
    # TODO implement other bit sizes
    # packed_values = 1
    # return packed_values.tobytes()


def get_back_values_from_buffer(bytes_: bytes):
    """
    Return the values from the given buffer

    This is for the encoded values in the neuroglancer segmentation format, so are in uint32.

    Parameters
    ----------
    bytes_ : bytes
        The buffer to get the values from

    Returns
    -------
    values : np.ndarray
        The values
    binary_representation : np.ndarray
        The binary representation of the values
    """
    values = np.frombuffer(bytes_, dtype=np.uint32).view(np.uint32)
    binary_representation = np.unpackbits(values.view(np.uint8), bitorder="little")
    return values, binary_representation


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
    buffer: bytearray,
    stored_lookup_tables: dict[bytes, tuple[int, int]],
    unique_values: da.Array,
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
    values_in_bytes = unique_values.tobytes()
    if values_in_bytes not in stored_lookup_tables:
        lookup_table_offset = _get_buffer_position(buffer)
        encoded_bits = _get_encoded_bits(unique_values)
        stored_lookup_tables[values_in_bytes] = (
            lookup_table_offset,
            encoded_bits,
        )
        buffer += values_in_bytes
    else:
        lookup_table_offset, encoded_bits = stored_lookup_tables[values_in_bytes]
    return lookup_table_offset, encoded_bits


def _create_encoded_values(
    buffer: bytearray, positions: da.Array, encoded_bits: int
) -> int:
    """Create the encoded values for the given values

    Parameters
    ----------
    buffer: bytearray
        The buffer to write the encoded values to
    positions: da.Array
        The values to encode (positions in the lookup table)
    encoded_bits: int
        The number of bits used to encode the values

    Returns
    -------
    encoded_values_offset: int
        The offset in the buffer to the encoded values
    """
    encoded_values_offset = _get_buffer_position(buffer)
    buffer += _pack_encoded_values(positions.compute(), encoded_bits)
    return encoded_values_offset


def create_segmentation_chunk(
    dask_data: da.Array,
    dimensions: tuple[tuple[int, int, int], tuple[int, int, int]],
    block_size: tuple[int, int, int] = (8, 8, 8),
):
    """Convert data in a dask array to a neuroglancer segmentation chunk"""
    bz, by, bx = block_size
    gz, gy, gx = get_grid_size_from_block_size(dask_data.shape, block_size)
    stored_lookup_tables: dict[bytes, tuple[int, int]] = {}
    # big enough to hold the 64-bit starting block headers
    buffer = bytearray(gx * gy * gz * 8)

    for z, y, x in np.ndindex(gz, gy, gx):
        block = dask_data[
            z * bz : (z + 1) * bz, y * by : (y + 1) * by, x * bx : (x + 1) * bx
        ]
        unique_values, indices = da.unique(block, return_inverse=True)
        # TODO mismatch between dimensions and data size after padding
        block = pad_block(block, block_size)

        lookup_table_offset, encoded_bits = _create_lookup_table(
            buffer, stored_lookup_tables, unique_values
        )
        encoded_values_offset = _create_encoded_values(buffer, indices, encoded_bits)
        block_offset = 8 * (x + gx * (y + gy * z))
        _create_block_header(
            buffer,
            lookup_table_offset,
            encoded_bits,
            encoded_values_offset,
            block_offset,
        )

    return Chunk(buffer, dimensions)
