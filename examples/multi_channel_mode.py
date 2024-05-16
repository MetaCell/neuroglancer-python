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
            names=["x", "y", "z", "c^"],
            units=["um", "um", "um", ""],
            scales=[0.26, 0.26, 0.29, 1.0],
        ),
    )
    state.layers["image"] = neuroglancer.ImageLayer(
        source=local_volume,
        volume_rendering_mode="on",
        volume_rendering_depth_samples=200,
        tool_bindings={
            "A": neuroglancer.VolumeRenderingTool(),
            "B": neuroglancer.VolumeRenderingDepthSamplesTool(),
        },
        panels=[add_render_panel()],
        **kwargs,
    )


def add_mesh_layer(state, **kwargs):
    transform = (
        neuroglancer.CoordinateSpaceTransform(
            # inputDimensions=neuroglancer.CoordinateSpace(
            #     names=["x", "y", "z"],
            #     units=["m", "m", "m"],
            #     scales=[2, 2, 2],
            # ),
            outputDimensions=neuroglancer.CoordinateSpace(
                names=["x", "y", "z"],
                units=["m", "m", "m"],
                scales=[2, 2, 2],
            ),
            matrix=np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ]
            ),
        ),
    )
    # TODO transform is not working
    mesh = (
        neuroglancer.SingleMeshLayer(
            source=neuroglancer.LayerDataSource(
                "vtk://https://storage.googleapis.com/neuroglancer-fafb-data/elmr-data/FAFB.surf.vtk.gz",
                # transform=transform,
            )
        )
    )
    state.layers["mesh"] = mesh


def get_shader():
    return """
#uicontrol invlerp normalized1(range=[0,20000], window=[0, 20000], clamp=true, channel=0)
#uicontrol invlerp normalized2(range=[0,20000], window=[0, 20000], clamp=true, channel=1)

void main() {
    float norm1 = normalized1();
    float norm2 = normalized2();
    emitIntensity(norm1 + norm2);
    emitGrayscale(norm1 + norm2);
}

"""


if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s, shader=get_shader())
    threedee_view(viewer)
    remove_axis_lines(viewer)
    show_statistics(viewer)
    update_title(viewer, "Multi-channel example")
    set_gpu_memory(viewer, gpu_memory=2)
    update_projection(viewer, orientation=[0.25, 0.6, 0.65, 0.3])
    open_browser(viewer, hang=False)
    # sleep(4)  # TODO this is a hack to wait for viewer to open
    # TODO can't at the moment as no trasnfer
    # update_projection(viewer, scale=455, depth=242)

"""
Here is another useful state

{
  "title": "Multi-channel example",
  "dimensions": {
    "x": [
      2.6e-7,
      "m"
    ],
    "y": [
      2.6e-7,
      "m"
    ],
    "z": [
      2.9e-7,
      "m"
    ]
  },
  "position": [
    129.119873046875,
    144.7524871826172,
    30.53437042236328
  ],
  "crossSectionScale": 1,
  "projectionOrientation": [
    -0.027111563831567764,
    0.018441088497638702,
    0.7081640362739563,
    0.7052861452102661
  ],
  "projectionScale": 256,
  "projectionDepth": -2.4893534183932,
  "layers": [
    {
      "type": "image",
      "source": "python://volume/34193c9b5ed717d01aad312fa7ef10a69ad55d52.7bb6f5bff724cdc35abc11319dd7f7589fe8e87d",
      "tab": "annotations",
      "panels": [
        {
          "side": "left",
          "row": 1,
          "size": 539,
          "tab": "rendering",
          "tabs": [
            "rendering",
            "source"
          ]
        }
      ],
      "shader": "#uicontrol invlerp normalized2(range=[0,20000], window=[0, 20000], clamp=true, channel=1)\n\nvoid main() {\n    float norm2 = normalized2();\n    emitGrayscale(norm2);\n}\n\n",
      "shaderControls": {
        "normalized2": {
          "window": [
            1430,
            12407
          ]
        }
      },
      "channelDimensions": {
        "c^": [
          1,
          ""
        ]
      },
      "volumeRenderingMode": "max",
      "volumeRenderingDepthSamples": 652.3555001881789,
      "name": "image"
    },
    {
      "type": "image",
      "source": "python://volume/34193c9b5ed717d01aad312fa7ef10a69ad55d52.7bb6f5bff724cdc35abc11319dd7f7589fe8e87d",
      "toolBindings": {
        "A": "volumeRenderingMode",
        "B": "volumeRenderingDepthSamples"
      },
      "tab": "annotations",
      "panels": [
        {
          "side": "left",
          "row": 2,
          "size": 539,
          "tab": "rendering",
          "tabs": [
            "rendering",
            "source"
          ]
        }
      ],
      "shader": "\n#uicontrol invlerp normalized1(range=[0,20000], window=[0, 20000], clamp=true, channel=0)\n\nvoid main() {\n    float norm1 = normalized1();\n    emitGrayscale(norm1);\n}\n\n",
      "channelDimensions": {
        "c^": [
          1,
          ""
        ]
      },
      "volumeRenderingMode": "on",
      "volumeRenderingDepthSamples": 652.3555001881789,
      "name": "image1"
    }
  ],
  "showAxisLines": false,
  "gpuMemoryLimit": 2000000000,
  "layout": "3d",
  "statistics": {
    "side": "left",
    "row": 3,
    "size": 539,
    "visible": true
  },
  "helpPanel": {
    "row": 4
  },
  "settingsPanel": {
    "row": 5
  }
}
"""