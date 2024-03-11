import argparse
import webbrowser
import neuroglancer
import numpy as np
import neuroglancer.cli


def add_image_layer(state):
    shape = (50,) * 3
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[40, 40, 40]
    )
    data = np.full(shape=shape, fill_value=255, dtype=np.uint8)
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers.append(
        name="image",
        layer=neuroglancer.ImageLayer(
            source=local_volume,
            volume_rendering_mode="ON",
            volume_rendering_depth_samples=400,
        ),
        shader="""
void main() {
    emitRGBA(vec4(1.0, 1.0, 1.0, 0.001));
    }
    """,
    )
    state.layout = "3d"


def get_shader():
    return """
#uicontrol invlerp normalized(range=[0,255], clamp=true)
#uicontrol vec3 color color(default="white")
void main() {
    float val = normalized();
    emitRGBA(vec4(color, 0.001));
    }
"""


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

    print(viewer.state.layers["image"].volumeRenderingGain)

    s1 = viewer.screenshot(size=[1000, 1000])
    print("shot 1 taken")

    with viewer.txn() as s:
        s.layers["image"].volume_rendering_gain = 10.0
    s2 = viewer.screenshot(size=[1000, 1000])
    print(viewer.state.layers["image"].volumeRenderingGain)
    print("shot 2 taken")

    with open("p1.png", "wb") as f:
        f.write(s1.screenshot.image)

    with open("p2.png", "wb") as f:
        f.write(s2.screenshot.image)

if __name__ == "__main__":
    main()
