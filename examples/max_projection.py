import argparse
import webbrowser
import neuroglancer
import numpy as np
import neuroglancer.cli

def create_volume():
    shape = (10, 40, 50)

    # Create the volume
    volume = np.zeros(shape)

    # Fill each row across the last dimension with 90 random data points and 10 data points that are 1s
    rng = np.random.default_rng()
    for i in range(shape[0]):
        for j in range(shape[1]):
            random_indices = rng.choice(shape[2], size=40, replace=False)
            volume[i, j, random_indices] = rng.random(40)
            volume[i, j, ~np.isin(np.arange(shape[2]), random_indices)] = 1
    return volume


def add_image_layer(state):
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[40, 40, 40]
    )
    local_volume = neuroglancer.LocalVolume(create_volume(), dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume, volume_rendering_mode="MAX", shader=get_shader()
    )
    state.layout = "3d"
    state.show_axis_lines = False


def get_shader():
    return """
#uicontrol invlerp normalized(range=[0,1])
void main() {
    emitGrayscale(normalized());
    }
"""


def launch_nglancer():
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    return viewer


if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s)
    webbrowser.open_new(viewer.get_viewer_url())

    s1 = viewer.screenshot(size=[100, 100])
    print("shot 1 taken")

    with open("p1.png", "wb") as f:
        f.write(s1.screenshot.image)