from cloudvolume import CloudVolume
import numpy as np

def create_float_cube(output_path="file://datasets/cube-float-res"):
    rawdata = np.full(shape=(100, 100, 100), fill_value=70.0).astype(np.float32)
    CloudVolume.from_numpy(
        rawdata,
        vol_path=output_path,
        resolution=(1.5, 1.5, 2.1),
        chunk_size=(100, 100, 100),
        layer_type="image",
        progress=True,
        compress=False
    )