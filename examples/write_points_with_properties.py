import argparse
import os
import shutil

import neuroglancer
import neuroglancer.cli
import neuroglancer.static_file_server
import neuroglancer.write_annotations
import numpy as np


def write_some_annotations(
    output_dir: str, coordinate_space: neuroglancer.CoordinateSpace
):
    properties = [
        neuroglancer.AnnotationPropertySpec(
            id="color",
            description="Color of the annotation",
            type="rgb",
            default="#ff0000",  # red
        ),
        neuroglancer.AnnotationPropertySpec(
            id="size",
            description="Size of the annotation",
            type="float32",
        ),
        neuroglancer.AnnotationPropertySpec(
            id="p_int8",
            type="int8",
            default=10,
        ),
        neuroglancer.AnnotationPropertySpec(
            id="p_uint8",
            type="uint8",
            default=10,
        ),
        neuroglancer.AnnotationPropertySpec(
            id="rgba_color",
            type="rgba",
            default="#00ff00ff",  # green with full opacity
        ),
        neuroglancer.AnnotationPropertySpec(
            id="p_enum1",
            type="uint16",
            default=0,
            enum_values=[0, 1, 2, 3],
            enum_labels=[
                "Option 0",
                "Option 1",
                "Option 2",
                "Option 3",
            ],
        ),
        neuroglancer.AnnotationPropertySpec(
            id="p_fnum32",
            type="float32",
            default=0.0,
            description="A float number property",
            enum_values=[0.0, 1.5, 2.6, 3.0],
            enum_labels=[
                "Zero",
                "One and a half",
                "Two point six",
                "Three",
            ],
        ),
        neuroglancer.AnnotationPropertySpec(
            id="p_boola",
            type="uint16",
            default=1,
            description="A boolean property",
            enum_values=[0, 1],
            enum_labels=["False", "True"],
        ),
    ]

    writer = neuroglancer.write_annotations.AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=properties,
    )

    writer.add_point(
        [20, 30, 40],
        color=(0, 255, 0),  # RGB color
        size=10,
        p_int8=1,
        p_uint8=2,
        rgba_color=(0, 255, 0, 255),  # RGBA color with full opacity
        p_enum1=1,
        p_fnum32=1.5,
    )
    writer.add_point(
        [50, 51, 52],
        color=(255, 0, 0),  # RGB color
        size=30,
        p_int8=2,
        p_uint8=3,
        rgba_color=(255, 0, 0, 255),  # RGBA color with full opacity
        p_enum1=2,
        p_fnum32=2.6,
    )
    writer.add_point(
        [40, 50, 20],
        color="#0000ff",  # RGB color
        size=20,
        p_int8=3,
        p_uint8=4,
        rgba_color=(0, 200, 255, 14),  # RGBA color with part opacity
        p_enum1=3,
        p_fnum32=3.0,
        p_boola=0,
    )
    writer.add_point([40, 50, 52])
    writer.write(os.path.join(output_dir, "point"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()

    tempdir = "/tmp/neuroglancer_annotations"
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)
    os.makedirs(tempdir)

    coordinate_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units=["nm", "nm", "nm"], scales=[10, 10, 10]
    )
    write_some_annotations(output_dir=tempdir, coordinate_space=coordinate_space)

    server = neuroglancer.static_file_server.StaticFileServer(
        static_dir=tempdir, bind_address=args.bind_address or "127.0.0.1", daemon=True
    )

    with viewer.txn() as s:
        s.layers["image"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=np.full(fill_value=200, shape=(100, 100, 100), dtype=np.uint8),
                dimensions=coordinate_space,
            ),
        )
        s.layers["points"] = neuroglancer.AnnotationLayer(
            source=f"precomputed://{server.url}/point",
            tab="rendering",
            shader="""
void main() {
  setColor(prop_color());
  setPointMarkerSize(prop_size());
}
        """,
        )
        s.selected_layer.layer = "points"
        s.selected_layer.visible = True
        s.show_slices = False
    print(viewer)
