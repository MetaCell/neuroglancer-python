# @license
# Copyright 2020 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        # volume_rendering_mode="off",
        panels=[add_render_panel()],
        volumeRenderingDepthSamples=512,
        tool_bindings={
            "A": neuroglancer.VolumeRenderingDepthSamplesTool(),
        }
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
