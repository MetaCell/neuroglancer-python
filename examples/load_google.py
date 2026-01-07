from cloudvolume import CloudVolume

def get_fafb_v14_voxels(
    locs_nm,
    input_voxel_size=(40, 4, 4),
    output_voxel_size=(40, 8, 8),
    size=(16, 160, 160),  # in voxel units
    precomputed_path="https://storage.googleapis.com/neuroglancer-fafb-data/fafb_v14/fafb_v14_orig/",
    mip=0
):
    vol = CloudVolume(precomputed_path, mip=mip, cache=True, parallel=True)

    # locs_vox = resample_locs(locs_nm, input_voxel_size, output_voxel_size)
    # size = daisy.Coordinate(size)
    shape = vol.shape  # (X, Y, Z)
    print("Volume shape:", shape)


if __name__ == "__main__":
    locs_nm = "example_location"  # Replace with actual location data
    get_fafb_v14_voxels(locs_nm)
    # You can add more functionality here, such as processing the volume or visualizing it.