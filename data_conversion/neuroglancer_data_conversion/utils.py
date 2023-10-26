from math import ceil

import dask.array as da


def pad_block(block: da.Array, block_size: tuple[int, int, int]) -> da.Array:
    """Pad the block to the given block size with zeros"""
    return da.pad(
        block,
        (
            (0, block_size[0] - block.shape[0]),
            (0, block_size[1] - block.shape[1]),
            (0, block_size[2] - block.shape[2]),
        ),
    )


def iterate_chunks(dask_data: da.Array):
    """Iterate over the chunks in the dask array"""
    chunk_layout = dask_data.chunks

    for zi, z in enumerate(chunk_layout[0]):
        for yi, y in enumerate(chunk_layout[1]):
            for xi, x in enumerate(chunk_layout[2]):
                chunk = dask_data.blocks[zi, yi, xi]

                # Calculate the chunk dimensions
                start = (
                    sum(chunk_layout[0][:zi]),
                    sum(chunk_layout[1][:yi]),
                    sum(chunk_layout[2][:xi]),
                )
                end = (start[0] + z, start[1] + y, start[2] + x)
                dimensions = (start, end)
                yield chunk, dimensions


def get_grid_size_from_block_size(
    data_shape: tuple[int, int, int], block_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    """
    Calculate the grid size from the block size

    Both the data shape and block size should be in z, y, x order
    
    Parameters
    ----------
    data_shape : tuple[int, int, int]
        The shape of the data
    block_size : tuple[int, int, int]
        The block size
    
    Returns
    -------
    tuple[int, int, int]
        The grid size as gz, gy, gx
    """
    gz = ceil(data_shape[0] / block_size[0])
    gy = ceil(data_shape[1] / block_size[1])
    gx = ceil(data_shape[2] / block_size[2])
    return gz, gy, gx