from cloudvolume import CloudVolume
from cloudvolume.lib import Vec


def create_cloudvolume_info(
    num_channels,
    data_type,
    num_mips,
    volume_size,
    chunk_size,
    output_path,
    allow_non_aligned_write=True,
):
    info = CloudVolume.create_new_info(
        num_channels=num_channels,
        layer_type="image",
        data_type=data_type,
        encoding="raw",
        resolution=[1, 1, 1],
        voxel_offset=[0, 0, 0],
        chunk_size=chunk_size,
        volume_size=volume_size,
        max_mip=num_mips - 1,
        factor=Vec(2, 2, 2),
    )

    vol = CloudVolume(
        "file://" + str(output_path),
        info=info,
        mip=0,
        non_aligned_writes=allow_non_aligned_write,
        fill_missing=True,
    )
    vol.commit_info()
    vol.provenance.description = "Example data conversion"
    vol.commit_provenance()

    del vol  # Close the volume

    # Create dirs for each MIP level
    vols = [
        CloudVolume(
            "file://" + str(output_path),
            mip=i,
            compress=False,
            non_aligned_writes=allow_non_aligned_write,
            fill_missing=True,
        )
        for i in range(num_mips)
    ]
    progress_dir = output_path / "progress"
    progress_dir.mkdir(exist_ok=True)
    
    return vols
