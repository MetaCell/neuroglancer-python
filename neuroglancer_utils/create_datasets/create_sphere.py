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

    return (sphere.astype(np.float32) * 255.0)


def create_volume(output_path="file://datasets/sphere"):
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type="float32",  # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso, crackle
        encoding="raw",
        resolution=[40, 40, 40],  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=[100, 100, 100],  # units are voxels
        volume_size=[100, 100, 100],  # e.g. a cubic millimeter dataset
    )
    vol = CloudVolume(output_path, info=info)
    vol.commit_info()
    rawdata = create_sphere_in_cube(100, 0.9)
    print(rawdata[30:40])
    vol[:, :, :] = rawdata[:, :, :]
