from cloudvolume import CloudVolume
import numpy as np


def create_sphere_in_cube(size, radius):
    shape = (size,) * 3
    # Create a grid of coordinates
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Equation of a sphere
    sphere = xx**2 + yy**2 + zz**2 <= radius**2

    sphere = sphere.astype(np.float32)
    random_values = np.random.rand(*shape) / 20.0
    where_zero = sphere < 0.1
    sphere[where_zero] = random_values[where_zero]
    sphere[0 : size // 10, :, :] = 0.5
    sphere[9 * size // 10 :, :, :] = 0.5

    return sphere.astype(np.float32) * 255.0


def create_volume(output_path="file://datasets/sphere"):
    rawdata = create_sphere_in_cube(100, 0.9)
    #rawdata = np.full(shape=(100, 100, 100), fill_value=255.0).astype(np.float32)
    CloudVolume.from_numpy(
        rawdata,
        vol_path=output_path,
        resolution=(40, 40, 40),
        chunk_size=(100, 100, 100),
        layer_type="image",
        progress=True,
        compress=False
    )
