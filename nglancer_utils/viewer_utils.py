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

def open_browser(viewer):
    print(viewer)
    webbrowser.open_new(viewer.get_viewer_url())
    input("Press Enter to continue...")
