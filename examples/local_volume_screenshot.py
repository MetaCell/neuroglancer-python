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

import zarr

import neuroglancer

from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
    threedee_view,
    update_projection,
    remove_axis_lines,
    show_statistics,
    update_title,
    set_gpu_memory,
)
from neuroglancer_utils.layer_utils import add_render_panel


def add_image_layer(state, **kwargs):
    shape = (2500, 2500, 250)
    data = zarr.full(shape=shape, fill_value=255, chunks=[157, 157, 32], dtype="uint8")
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[40, 40, 400]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering_mode="ON",
        **kwargs,
    )

def get_shader():
    return """
#uicontrol invlerp normalized(range=[0,255], clamp=true)
#uicontrol vec3 color color(default="white")
void main() {
    float val = normalized();
    emitRGBA(vec4(color, val));
    }
    """

if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s, shader=get_shader())
    #threedee_view(viewer)
    remove_axis_lines(viewer)
    update_title(viewer, "Volume control example")
    set_gpu_memory(viewer, gpu_memory=2)
    update_projection(viewer, orientation=[0.25, 0.6, 0.65, 0.3])
    open_browser(viewer, hang=False)
    sleep(4) # TODO this is a hack to wait for viewer to open
    update_projection(viewer, scale=4555, depth=4429)

    s1 = viewer.screenshot(size=[1000, 1000])
    print(s1)

    with viewer.txn() as s:
        s.layers["image"].volume_rendering_gain = 1.0
    s2 = viewer.screenshot(size=[1000, 1000])
    print(s2)

    with open("p1.png", "wb") as f:
        f.write(s1.screenshot.image)

    with open("p2.png", "wb") as f:
        f.write(s2.screenshot.image)
