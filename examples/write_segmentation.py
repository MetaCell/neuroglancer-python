from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path

def load_data(filename):
    url = parse_url(filename)
    reader = Reader(url)
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data[0]
    return dask_data


def write_segmentation():
    pass


def main(filename):
    dask_data = load_data(filename)


if __name__ == "__main__":
    base_directory = Path("/media/starfish/LargeSSD/data/cryoET/data/segmentation")
    filename = base_directory / "00004_actin_ground_truth_zarr"
    main(filename)