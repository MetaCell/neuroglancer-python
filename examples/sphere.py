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
import numpy as np
import zarr

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


def create_sphere_in_cube(size, radius):
    shape = (size,) * 3
    # Create a grid of coordinates
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Equation of a sphere
    sphere = xx**2 + yy**2 + zz**2 <= radius**2

    sphere = sphere.astype(np.float32)
    random_values = np.random.rand(*shape) / 20.0
    where_zero = sphere < 0.1
    sphere[where_zero] = random_values[where_zero]
    sphere[0:size // 10, :, :] = 0.5

    return sphere.astype(np.float32)


def add_image_layer(state, **kwargs):
    array = create_sphere_in_cube(500, 0.6) * 255
    data = zarr.array(array)
    print(data.info)
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[400, 400, 400]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
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
