import neuroglancer

from neuroglancer_utils.create_datasets.create_mc_logo import create_mc_logo
from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
)

from time import sleep


def add_image_layer(state):
    data = create_mc_logo()
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        shader="""
#uicontrol invlerp normalized
void main() {
    emitRGBA(vec4(0.0, 0.0, normalized(), normalized()));
    }
""",
    )
    state.show_axis_lines = False


if __name__ == "__main__":
    viewer = launch_nglancer()
    open_browser(viewer, hang=False)
    sleep(2)

    with viewer.txn() as s:
        add_image_layer(s)
