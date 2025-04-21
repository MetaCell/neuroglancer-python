from cloudvolume import CloudVolume
import numpy as np


def create_multi(output_path="file://datasets/multic"):
    generator = np.random.default_rng(0)
    data = generator.random((40, 30, 20, 8)) * 10000
    CloudVolume.from_numpy(
        data.astype(np.float32),
        vol_path=output_path,
        resolution=(40, 40, 40),
        chunk_size=(10, 10, 10),
        layer_type="image",
        progress=True,
        compress=False,
    )

if __name__ == "__main__":
    create_multi()