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

import numpy as np

import neuroglancer

from nglancer_utils.viewer_utils import launch_nglancer, open_browser


def add_image_layer(state):
    shape = (128, 128, 128)
    data = np.ones(shape)
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="cm", scales=[1, 1, 1]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering=True,
        tool_bindings={
            "A": neuroglancer.VolumeRenderingTool(),
        },
    )


if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s)
    open_browser(viewer)
