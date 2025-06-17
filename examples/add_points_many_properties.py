import argparse

import neuroglancer
import neuroglancer.cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    # We need first, RGB
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
            default=10,
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
    ]

    def random_values(i):
        return [i * t for t in range(300)]

    with viewer.txn() as s:
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y"], units="nm", scales=[1, 1]
        )
        s.position = [150, 150]
        s.layers.append(
            name="a",
            layer=neuroglancer.LocalAnnotationLayer(
                dimensions=s.dimensions,
                annotation_properties=properties,
                annotations=[
                    neuroglancer.PointAnnotation(
                        id="1",
                        point=[150, 150],
                        props=[
                            "#00ff00",  # green
                            5,  # size
                            6,  # p_int8
                            7,  # p_uint8
                            "#ff00ffaa",  # rgba_color
                            2,  # p_enum1 (Option 2)
                            2.6,  # p_fnum32 (Two point six)
                        ],
                    ),
                    neuroglancer.PointAnnotation(
                        id="2",
                        point=[250, 100],
                        props=[
                            "#0000ff",  # blue
                            10,  # size
                            -5,  # p_int8
                            20,  # p_uint8
                            "#00ffffff",  # yellow with full opacity
                            1,  # p_enum1 (Option 1)
                            1.5,  # p_fnum32 (One and a half)
                        ],
                    ),
                    neuroglancer.PointAnnotation(
                        id="3",
                        point=[20, 20],
                    ),
                ],
                shader="""
void main() {
  setColor(prop_color());
  setPointMarkerSize(prop_size());
}
""",
            ),
        )
        s.layout = "xy"
        s.selected_layer.layer = "a"
    print("Use `Control+right click` to display annotation details.")
    print(viewer)
