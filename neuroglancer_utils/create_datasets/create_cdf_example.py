from cloudvolume import CloudVolume
import numpy as np


def create_cdf_example(output_path="file://datasets/cdf"):
    shape = (40,) * 3
    data = np.zeros(shape=shape, dtype=np.uint8)
    data[:10] = 0
    data[10:20] = 1
    data[20:30] = 2
    data[30:] = 3
    CloudVolume.from_numpy(
        data,
        vol_path=output_path,
        resolution=(40, 40, 40),
        chunk_size=(40, 40, 40),
        layer_type="image",
        progress=True,
        compress=False,
    )
