import neuroglancer
import numpy as np

from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
)

from time import sleep


def add_image_layer(state):
    data = np.full(shape=(10,) * 3, fill_value=255, dtype=np.uint8)
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        shader="""
void main() {
    emitRGBA(vec4(1.0, 1.0, 1.0, 0.001));
    }
    """,
    )
    state.show_axis_lines = False
    state.projection_scale = 1e-8
    state.position = [5, 5, 5]
    state.showSlices = True
    state.hideCrossSectionBackground3D = True


if __name__ == "__main__":
    viewer = launch_nglancer()
    open_browser(viewer, hang=False)
    sleep(2)

    with viewer.txn() as s:
        add_image_layer(s)
