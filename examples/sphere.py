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
from neuroglancer_utils.create_datasets.create_sphere import create_sphere
from neuroglancer_utils.local_server import create_server

from neuroglancer_utils.layer_utils import add_render_panel
from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
    remove_axis_lines,
    set_gpu_memory,
    threedee_view,
    update_projection,
    update_title,
)

def add_image_layer(state, **kwargs):
    state.layers["image"] = neuroglancer.ImageLayer(
        source="precomputed://http://127.0.0.1:9000/sphere",
        volume_rendering_mode="max",
        tool_bindings={
            "A": neuroglancer.VolumeRenderingGainTool(),
        },
        panels=[add_render_panel(side="right")],
        **kwargs,
    )


def get_shader():
    return """#uicontrol invlerp normalized(range=[0,255], clamp=true)
void main() {
    float val = normalized();
    emitIntensity(val);
    emitGrayscale(val);
}
"""


if __name__ == "__main__":
    create_sphere()
    create_server(directory="datasets")
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s, shader=get_shader())
    threedee_view(viewer)
    remove_axis_lines(viewer)
    update_title(viewer, "Sphere example")
    set_gpu_memory(viewer, gpu_memory=2)
    update_projection(viewer, orientation=[0.25, 0.6, 0.65, 0.3])
    open_browser(viewer, hang=False)
    sleep(4)  # TODO this is a hack to wait for viewer to open
    update_projection(viewer, scale=1500, depth=3000)
