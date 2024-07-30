import argparse
import webbrowser
import neuroglancer
import numpy as np
import neuroglancer.cli


def add_image_layer(state):
    shape = (40,) * 3
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[40, 40, 40]
    )
    data = np.zeros(shape=shape, dtype=np.uint8)
    data[:10] = 0
    data[10:20] = 1
    data[20:30] = 2
    data[30:] = 3
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers.append(
        name="image",
        layer=neuroglancer.ImageLayer(
            source=local_volume,
            volume_rendering_mode="max",
            volume_rendering_depth_samples=40,
        ),
        shader="""
#uicontrol invlerp normalized(range=[1,2])
void main() {
  emitGrayscale(normalized());
    }
    """,
    )


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
