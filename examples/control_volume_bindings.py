import neuroglancer
from nglancer_utils.viewer_utils import (
    launch_nglancer,
    generic_volume_setup,
    open_browser,
)
from nglancer_utils.layer_utils import add_render_panel

if __name__ == "__main__":
    viewer = launch_nglancer()
    with viewer.txn() as s:
        s.layers["image"] = neuroglancer.ImageLayer(
            source="precomputed://gs://neuroglancer-public-data/flyem_fib-25/image",
            tool_bindings={
                "A": neuroglancer.VolumeRenderingTool(),
                "B": neuroglancer.VolumeRenderingSamplesPerRayTool(),
            },
            panels=[add_render_panel()],
        )
    generic_volume_setup(viewer)
    open_browser(viewer)
