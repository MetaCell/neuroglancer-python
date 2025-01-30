from cloudvolume import CloudVolume
import numpy as np


def create_cube(output_path="file://datasets/cube-sixteen"):
    rawdata = np.full(shape=(100, 100, 100), fill_value=70).astype(np.uint32)
    CloudVolume.from_numpy(
        rawdata,
        vol_path=output_path,
        resolution=(40, 40, 40),
        chunk_size=(100, 100, 100),
        layer_type="image",
        progress=True,
        compress=False,
    )
