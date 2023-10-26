from typing import Any
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path
from tqdm import tqdm
import json

import numpy as np
import dask.array as da

from neuroglancer_data_conversion.utils import iterate_chunks
from neuroglancer_data_conversion.compressed_segmentation_encoding import (
    create_segmentation_chunk,
)


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
    data_directory: str,
):
    """Create the metadata for the segmentation"""
    metadata = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint32",
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [list(chunk_size)],
                "encoding": "compressed_segmentation",
                "compressed_segmentation_block_size": list(block_size),
                # TODO resolution is in nm, while for others there is no units
                "resolution": [1, 1, 1],
                "key": data_directory,
                "size": list(data_size),
            }
        ],
        "type": "segmentation",
    }
    return metadata


def write_metadata(metadata: dict[str, Any], output_directory: Path):
    """Write the segmentation to the given directory"""
    metadata_path = output_directory / "info"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def create_segmentation(dask_data: da.Array, block_size):
    """Yield the neuroglancer segmentation format chunks"""
    to_iterate = iterate_chunks(dask_data)
    num_iters = np.prod(dask_data.numblocks)
    for chunk, dimensions in tqdm(
        to_iterate, desc="Processing chunks", total=num_iters
    ):
        yield create_segmentation_chunk(chunk, dimensions, block_size)


def main(filename, block_size=(64, 64, 64), data_directory="data"):
    """Convert the given OME-Zarr file to neuroglancer segmentation format with the given block size"""
    print(f"Converting {filename} to neuroglancer compressed segmentation format")
    dask_data = load_data(filename)
    output_directory = filename.parent / f"precomputed-{filename.stem[:-5]}"
    output_directory.mkdir(parents=True, exist_ok=True)
    for c in create_segmentation(dask_data, block_size):
        c.write_to_directory(output_directory / data_directory)

    metadata = _create_metadata(
        dask_data.chunksize, block_size, dask_data.shape, data_directory
    )
    write_metadata(metadata, output_directory)
    print(f"Wrote segmentation to {output_directory}")


if __name__ == "__main__":
    base_directory = Path("/media/starfish/LargeSSD/data/cryoET/data")
    actin_filename = base_directory / "00004_actin_ground_truth_zarr"
    microtubules_filename = base_directory / "00004_MT_ground_truth_zarr"
    block_size = (32, 32, 32)
    main(actin_filename, block_size)
    main(microtubules_filename, block_size)
