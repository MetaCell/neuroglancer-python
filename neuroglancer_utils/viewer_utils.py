import argparse
import webbrowser

import neuroglancer
import neuroglancer.cli


def launch_nglancer():
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()

    return viewer


def open_browser(viewer, hang=True):
    print(viewer)
    webbrowser.open_new(viewer.get_viewer_url())
    if hang:
        input("Press Enter to continue...")


def generic_volume_setup(viewer):
    threedee_view(viewer)
    remove_axis_lines(viewer)
    show_statistics(viewer)
    set_gpu_memory(viewer, gpu_memory=2)
    update_title(viewer, "Volume example")

def threedee_view(viewer):
    with viewer.txn() as state:
        state.layout = "3d"


def update_projection(viewer, orientation=None, scale=None, depth=None):
    with viewer.txn() as state:
        if orientation is not None:
            state.projection_orientation = orientation
        if scale is not None:
            state.projection_scale = scale
        if depth is not None:
            state.projection_depth = depth


def remove_axis_lines(viewer):
    with viewer.txn() as state:
        state.show_axis_lines = False


def show_statistics(viewer, side="left", row=1, col=0):
    with viewer.txn() as state:
        state.statistics = neuroglancer.StatisticsDisplayState(
            side=side, row=row, col=col, visible=True
        )


def update_title(viewer, title):
    with viewer.txn() as state:
        state.title = title


def set_gpu_memory(viewer, gpu_memory=2):
    """Default is 2GB here"""
    with viewer.txn() as state:
        state.gpu_memory_limit = int(gpu_memory * (10 ** 9))
