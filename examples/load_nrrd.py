import argparse
import webbrowser
import neuroglancer
import neuroglancer.cli
import nrrd
from pathlib import Path
from cloudvolume import CloudVolume

HERE = Path(__file__).parent

NAMES = ["12vj", "1567", "101b"]
PATHS = [HERE / f"{name}.nrrd" for name in NAMES]
OUTPUT_PATHS = [f"file://datasets/{name}" for name in NAMES]


def convert_to_precomputed(nrrd_path, output_path):
    readdata, header = nrrd.read(nrrd_path)
    # Cloud volume expects resolution in nm - but given in um
    scales = [header["space directions"][i][i] * 1000 for i in range(3)]
    CloudVolume.from_numpy(
        readdata,
        vol_path=output_path,
        chunk_size=(256, 256, 128),
        resolution=scales,
        layer_type="image",
        progress=True,
        compress=False,
    )


def add_image_layer(state, path, name="image"):
    readdata, header = nrrd.read(path)
    scales = [header["space directions"][i][i] for i in range(3)]
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="um", scales=scales
    )
    data = readdata
    local_volume = neuroglancer.LocalVolume(data, dimensions)
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
        for i in range(3):
            convert_to_precomputed(PATHS[i], OUTPUT_PATHS[i])
            path = PATHS[i]
            name = NAMES[i]
            add_image_layer(s, path, name)
    exit(-1)
    webbrowser.open_new(viewer.get_viewer_url())


if __name__ == "__main__":
    main()
