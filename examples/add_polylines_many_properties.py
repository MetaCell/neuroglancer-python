import argparse

import neuroglancer
import neuroglancer.cli

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    # Make lot of properties to test the performance
    lots_of_properties = [
        neuroglancer.AnnotationPropertySpec(
            id=f"random_{t}",
            type="float32",
            default=10,
        )
        for t in range(300)
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
                annotation_properties=[
                    neuroglancer.AnnotationPropertySpec(
                        id="color",
                        type="rgb",
                        default="red",
                    ),
                    neuroglancer.AnnotationPropertySpec(
                        id="size",
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
                    *lots_of_properties,
                ],
                annotations=[
                    neuroglancer.PolyLineAnnotation(
                        id="1",
                        points=[[150, 150], [200, 100], [30, 40]],
                        props=["#0f0", 5, 6, 7, *random_values(1)],
                    ),
                    neuroglancer.PolyLineAnnotation(
                        id="2",
                        points=[
                            [250, 100],
                            [20, 20],
                            [32, 42],
                            [50, 60],
                            [70, 80],
                            [90, 100],
                        ],
                        props=["#ff0", 30, 7, 9, *random_values(2)],
                    ),
                ],
                shader="""
void main() {
  setColor(prop_color());
  setPolyEndpointMarkerSize(prop_size());
}
""",
            ),
        )
        s.layout = "xy"
        s.selected_layer.layer = "a"
    print("Use `Control+right click` to display annotation details.")
    print(viewer)
