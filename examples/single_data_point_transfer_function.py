import neuroglancer
import numpy as np

from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
)
from neuroglancer_utils.layer_utils import add_render_panel

from time import sleep


def colormap_version(viewer):
    shader = """
#uicontrol invlerp normalized
#uicontrol transferFunction colormap
void main() {
    emitRGBA(colormap());
}
"""
    shaderControls = {
        "normalized": {
            "range": [0, 100],
            "window": [0, 100],
            "channel": [],
        },
        "colormap": {
            "controlPoints": [[0, "#000000", 0.0], [84, "#ffffff", 1.0]],
            "window": [0, 100],
            "channel": [],
            "defaultColor": "#ff00ff",
        },
    }
    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y"], units="nm", scales=[1, 1]
        )
        s.position = [0.5, 0.5]
        s.layers.append(
            name="image",
            layer=neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    dimensions=s.dimensions,
                    data=np.full(shape=(1, 1), dtype=np.uint32, fill_value=42),
                ),
                panels=[add_render_panel()],
            ),
            visible=True,
            shader=shader,
            shader_controls=shaderControls,
            opacity=1.0,
            blend="additive",
        )
        s.layout = "xy"
        s.cross_section_scale = 1e-6
        s.show_axis_lines = False


if __name__ == "__main__":
    viewer = launch_nglancer()
    open_browser(viewer, hang=False)
    sleep(2)
    colormap_version(viewer)

    with viewer.txn() as s:
        layer = s.layers[0]
        print(layer.shader_controls["normalized"])
        print(layer.shader_controls["colormap"])

    # inp = input("Press enter when ready to change the shader controls...")
    # with viewer.txn() as s:
    #     layer = s.layers[0]
    #     print(layer.shader_controls["colormap"])
    #     layer.shader_controls = {
    #         "colormap": neuroglancer.TransferFunctionParameters(
    #             range=[0, 100],
    #             controlPoints=[
    #                 {"input": 10, "color": "#0f00ff", "opacity": 0.4},
    #                 {"input": 150, "color": "#ff00ff", "opacity": 0.1},
    #             ],
    #             channel=[],
    #             color="#ff0000",
    #         )
    #     }
    #     print(layer.shader_controls["colormap"])
