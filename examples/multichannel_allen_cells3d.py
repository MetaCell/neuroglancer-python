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

from skimage.data import cells3d
import numpy as np
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
    data = cells3d()
    data = np.moveaxis(data, [0, 1, 2, 3], [2, 3, 1, 0])
    local_volume = neuroglancer.LocalVolume(
        data=data,
        dimensions=neuroglancer.CoordinateSpace(
            names=["x", "y", "z", "c"],
            units=["um", "um", "um", "s"],
            scales=[0.26, 0.26, 0.29, 1.0],
        ),
    )
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        tool_bindings={
            "A": neuroglancer.VolumeRenderingModeTool(),
            "B": neuroglancer.VolumeRenderingSamplesPerRayTool(),
        },
        panels=[add_render_panel()],
        **kwargs,
    )


def get_shader():
    return """
#uicontrol float gain slider(min=0, max=10, default=1.0)
#uicontrol invlerp normalized(range=[0,65535], clamp=true)
#uicontrol vec3 color color(default="white")
void main() {
    float val = normalized();
    emitRGBA(vec4(color, val * gain));
    }
    """


if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s, shader=get_shader())
    threedee_view(viewer)
    remove_axis_lines(viewer)
    show_statistics(viewer)
    update_title(viewer, "Volume control example")
    set_gpu_memory(viewer, gpu_memory=2)
    update_projection(viewer, orientation=[0.25, 0.6, 0.65, 0.3])
    open_browser(viewer, hang=False)
    sleep(4)  # TODO this is a hack to wait for viewer to open
    update_projection(viewer, scale=455, depth=242)
