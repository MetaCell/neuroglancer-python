import argparse
import webbrowser
import neuroglancer
import neuroglancer.cli
import nrrd
from pathlib import Path

HERE = Path(__file__).parent
NRRD_FILE_PATH = HERE / "volume1.nrrd"


def add_image_layer(state):
    readdata, header = nrrd.read(NRRD_FILE_PATH)
    scales = [header["space directions"][i][i] for i in range(3)]
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="um", scales=scales
    )
    data = readdata
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers.append(
        name="image",
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
        add_image_layer(s)
    webbrowser.open_new(viewer.get_viewer_url())


if __name__ == "__main__":
    main()
