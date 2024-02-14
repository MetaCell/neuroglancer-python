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

DATA_TYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "float32",
]

def add_image_layer(state, dtype, **kwargs):
    shape = (2500, 2500, 250)
    data = zarr.full(shape=shape, fill_value=70, chunks=[157, 157, 32], dtype=dtype)
    print(data.info)
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[40, 40, 400]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers[f"image {dtype}"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        # tool_bindings={
        #     "A": neuroglancer.VolumeRenderingModeTool(),
        # },
        panels=[add_render_panel()],
        **kwargs,
    )

def get_shader():
    return """
#uicontrol transferFunction colormap(range=[0,128])
void main() {
    emitRGBA(colormap());
    }
    """

if __name__ == "__main__":
    viewer = launch_nglancer()
    for dtype in DATA_TYPES:
        with viewer.txn() as s:
            add_image_layer(s, dtype, shader=get_shader())
    threedee_view(viewer)
    remove_axis_lines(viewer)
    show_statistics(viewer)
    update_title(viewer, "Volume control example")
    set_gpu_memory(viewer, gpu_memory=2)
    update_projection(viewer, orientation=[0.25, 0.6, 0.65, 0.3])
    open_browser(viewer, hang=False)
    sleep(4) # TODO this is a hack to wait for viewer to open
    update_projection(viewer, scale=4555, depth=4429)
