import argparse
import itertools
import webbrowser
import neuroglancer
import neuroglancer.cli
from pathlib import Path
from bfio import BioReader
import dask
import dask.array
import numpy as np

HERE = Path(__file__).parent

FILEPATH = Path("x.ome.tiff")


def add_image_layer(state, path, name="image"):
    br = BioReader(str(path), backend="bioformats")
    chunk_shape = np.array([256, 256, 128, 1])
    shape = np.array(br.shape)
    num_chunks_per_dim = np.ceil(shape / chunk_shape).astype(int)
    padded_chunk_shape = num_chunks_per_dim * chunk_shape

    def chunked_reader(x_i, y_i, z_i, c):
        x_start, x_end = x_i * chunk_shape[0], min((x_i + 1) * chunk_shape[0], shape[0])
        y_start, y_end = y_i * chunk_shape[1], min((y_i + 1) * chunk_shape[1], shape[1])
        z_start, z_end = z_i * chunk_shape[2], min((z_i + 1) * chunk_shape[2], shape[2])

        # Read the chunk from the BioReader
        chunk = br.read(
            X=(x_start, x_end), Y=(y_start, y_end), Z=(z_start, z_end), C=(c,)
        )
        # Extend the chunk to be X, Y, Z, 1 not just X, Y, Z
        chunk = np.expand_dims(chunk, axis=-1)
        # If the chunk is smaller than the padded chunk shape, pad it
        if chunk.shape != tuple(chunk_shape[:3]):
            padded_chunk = np.zeros(chunk_shape, dtype=chunk.dtype)
            padded_chunk[: chunk.shape[0], : chunk.shape[1], : chunk.shape[2], :] = (
                chunk
            )
            return padded_chunk
        return chunk

    def chunk_size(x_i, y_i, z_i, c):
        x_start, x_end = x_i * chunk_shape[0], min((x_i + 1) * chunk_shape[0], shape[0])
        y_start, y_end = y_i * chunk_shape[1], min((y_i + 1) * chunk_shape[1], shape[1])
        z_start, z_end = z_i * chunk_shape[2], min((z_i + 1) * chunk_shape[2], shape[2])

        return (x_end - x_start, y_end - y_start, z_end - z_start, 1)

    lazy_reader = dask.delayed(chunked_reader)
    lazy_chunks = [
        lazy_reader(x, y, z, c)
        for x, y, z, c in itertools.product(*[range(i) for i in num_chunks_per_dim])
    ]
    # chunk_sizes = [
    #     chunk_size(x, y, z, c)
    #     for x, y, z, c in itertools.product(*[range(i) for i in num_chunks_per_dim])
    # ]
    sample = lazy_chunks[
        0
    ].compute()  # load the first chunk (assume rest are same shape/dtype)
    arrays = [
        dask.array.from_delayed(lazy_chunk, dtype=sample.dtype, shape=sample.shape)
        for lazy_chunk in lazy_chunks
    ]
    x = dask.array.concatenate(arrays)
    print(x.shape, shape, np.prod(x.shape), np.prod(padded_chunk_shape))
    # We need to reshape in iterations, 
    # x.reshape(padded_chunk_shape)
    scales = [1, 1, 1, 1]
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "c"], units="um", scales=scales
    )
    local_volume = neuroglancer.LocalVolume(x, dimensions)
    state.layers.append(
        name=name,
        layer=neuroglancer.ImageLayer(
            source=local_volume,
            volume_rendering_mode="ON",
            volume_rendering_depth_samples=400,
        ),
        shader="""
#uicontrol invlerp normalized
void main() {
    float val = normalized();
    emitRGBA(vec4(val, val, val, val));
    }
    """,
    )
    state.layout = "3d"


def launch_nglancer():
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    return viewer


def main():
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s, FILEPATH, "image")
    webbrowser.open_new(viewer.get_viewer_url())


if __name__ == "__main__":
    main()
