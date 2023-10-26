import struct
import numpy as np

from neuroglancer_data_conversion.utils import get_grid_size_from_block_size
from neuroglancer_data_conversion.chunk import Chunk

# The units for the offsets are in 32-bit words
OFFSET_BYTES = 4
LEAST_SIGNIFICANT_24_BITS = 0x00FFFFFF
ALLOWED_ENCODED_BITS = (0, 1, 2, 4, 8, 16, 32)


def _verify_bits(encoded_bits: int) -> int:
    if encoded_bits not in ALLOWED_ENCODED_BITS:
        raise ValueError(
            f"The encoded bits must one of {ALLOWED_ENCODED_BITS} but got {encoded_bits}"
        )
    return encoded_bits


def _verify_encoded_values_offset(
    encoded_values_offset: int, encoded_bits: int, chunk_size: int
) -> int:
    if encoded_bits != 0 and encoded_values_offset > chunk_size:
        raise ValueError(
            f"The encoded values offset must be less than the chunk length but got {encoded_values_offset} and {chunk_size}"
        )
    return encoded_values_offset

def _verify_lookup_table_offset(lookup_table_offset: int, chunk_size: int) -> int:
    if lookup_table_offset > chunk_size:
        raise ValueError(
            f"The lookup table offset must be less than the chunk length but got {lookup_table_offset} and {chunk_size}"
        )
    return lookup_table_offset


def _decode_block_header(header: tuple[int, int], chunk_size: int) -> tuple[int, int, int]:
    first_int, second_int = header
    # Pull the lookup offset from the least significant 24 bits
    lookup_table_offset = _verify_lookup_table_offset(
        OFFSET_BYTES * (first_int & LEAST_SIGNIFICANT_24_BITS), chunk_size
    )

    # Pull the encoded bits from the most significant 8 bits
    encoded_bits = _verify_bits(first_int >> 24)

    # Pull the encoded values offset from the second integer
    encoded_values_offset = _verify_encoded_values_offset(OFFSET_BYTES * second_int, encoded_bits, chunk_size)

    return lookup_table_offset, encoded_bits, encoded_values_offset


def _decode_block(block: np.ndarray, block_offset: int, chunk_size: int) -> np.ndarray:
    # Read the block header
    header = struct.unpack_from("<II", block, block_offset)
    lookup_table_offset, encoded_bits, encoded_values_offset = _decode_block_header(header, chunk_size)


def decode_chunk(chunk: Chunk, block_size) -> np.ndarray:
    """Decode the given chunk

    Parameters
    ----------
    chunk : np.ndarray
        The chunk to decode
    encoded_bits : int
        The number of bits used to encode the chunk

    Returns
    -------
    np.ndarray
        The decoded chunk
    """
    gz, gy, gx = get_grid_size_from_block_size(chunk.shape, block_size)

    for z, y, x in np.ndindex(gz, gy, gx):
        block_offset = 8 * (x + gx * (y + gy * z))
        chunk[z, y, x] = _decode_block(chunk, block_offset, chunk_size=chunk.size)
