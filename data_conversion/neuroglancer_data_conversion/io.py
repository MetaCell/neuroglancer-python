from typing import Any
import json
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path
import dask.array as da

def load_omezarr_data(input_filepath: Path) -> da.Array:
    """Load the OME-Zarr data and return a dask array"""
    url = parse_url(input_filepath)
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[0]
    return dask_data.persist()


def write_metadata(metadata: dict[str, Any], output_directory: Path):
    """Write the segmentation to the given directory"""
    metadata_path = output_directory / "info"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)