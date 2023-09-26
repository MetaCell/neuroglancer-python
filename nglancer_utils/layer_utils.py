import neuroglancer


def add_render_panel(side="left", row=0, col=0):
    return neuroglancer.LayerSidePanelState(
        side=side,
        col=col,
        row=row,
        tab="render",
        tabs=["rendering", "source"],
    )
