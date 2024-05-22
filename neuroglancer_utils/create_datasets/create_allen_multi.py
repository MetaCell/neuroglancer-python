from skimage.data import cells3d

from cloudvolume import CloudVolume
import numpy as np


def create_allen_multi(output_path="file://datasets/allen"):
    data = cells3d()
    data = np.moveaxis(data, [0, 1, 2, 3], [2, 3, 1, 0])
    CloudVolume.from_numpy(
        data,
        vol_path=output_path,
        resolution=(40, 40, 40),
        chunk_size=(128, 128, 64),
        layer_type="image",
        progress=True,
        compress=False,
    )
