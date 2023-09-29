from time import sleep

import neuroglancer
from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
    update_projection,
    generic_volume_setup,
)
from neuroglancer_utils.layer_utils import add_render_panel
from neuroglancer_utils.shaders import volume_rendering_shader

URL = r"zarr://s3://aind-open-data/exaSPIM_609281_2022-11-03_13-49-18_stitched_2022-11-22_12-07-00/fused.zarr/fused.zarr/"


def add_image_layer(state):
    state.layers["brain"] = neuroglancer.ImageLayer(
        source=URL,
        shader=volume_rendering_shader,
        volume_rendering=True,
        panels=[add_render_panel()],
    )


def make_cordinate_space(state):
    co_ords = neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "t"],
        units=["nm", "nm", "um", "ms"],
        scales=[748, 748, 1, 1],
    )
    state.dimensions = co_ords


if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s)
        make_cordinate_space(s)
    generic_volume_setup(viewer)
    open_browser(viewer, hang=False)
    sleep(4)
    update_projection(viewer, orientation=[0, 0, 0, 1], scale=50000, depth=-0.12)
    # get shader value
    # s.layers["brain"].shader_controls["gain"]
