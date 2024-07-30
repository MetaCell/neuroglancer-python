from cloudvolume import CloudVolume
import numpy as np


def create_border_example(output_path="file://datasets/border"):
    shape = (40,) * 3
    data = np.zeros(shape=shape, dtype=np.uint8)
    data[4:36, 4:36, 4:36] = 10
    CloudVolume.from_numpy(
        data,
        vol_path=output_path,
        resolution=(40, 40, 40),
        chunk_size=(32, 32, 32),
        layer_type="image",
        progress=True,
        compress=False,
    )
