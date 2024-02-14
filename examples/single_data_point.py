import neuroglancer
import numpy as np

from neuroglancer_utils.viewer_utils import (
    launch_nglancer,
    open_browser,
)

from time import sleep


def setup_single_data_point_viewer(viewer, shader_controls, shader=None):
    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y"], units="nm", scales=[1, 1]
        )
        s.position = [0.5, 0.5]
        layer_kwargs = {
            "name": "image",
            "layer": neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    dimensions=s.dimensions,
                    data=np.full(shape=(1, 1), dtype=np.uint32, fill_value=42),
                ),
            ),
            "visible": True,
            "shader_controls": shader_controls,
        }
        if shader is not None:
            layer_kwargs["shader"] = shader
        s.layers.append(**layer_kwargs)
        s.layout = "xy"
        s.cross_section_scale = 1e-6
        s.show_axis_lines = False


def normalized_version(viewer):
    setup_single_data_point_viewer(
        viewer,
        {
            "normalized": {
                "range": [0, 42],
            },
        },
    )


def colormap_version(viewer):
    shader = """
#uicontrol transferFunction colormap
void main() {
    emitRGBA(colormap());
}
"""
    shaderControls = {
        "colormap": {
            "controlPoints": [[0, "#000000", 0.0], [84, "#ffffff", 1.0]],
            "range": [0, 100],
            "channel": [],
            "color": "#ff00ff",
        }
    }
    setup_single_data_point_viewer(viewer, shaderControls, shader)


if __name__ == "__main__":
    viewer = launch_nglancer()
    open_browser(viewer, hang=False)
    sleep(2)
    colormap_version(viewer)
