import argparse
import json
import ndjson
from pathlib import Path

import neuroglancer
import neuroglancer.write_annotations
import neuroglancer.static_file_server
import neuroglancer.cli


def load_data(metadata_path, annotations_path):
    """Load in the metadata (json) and annotations (ndjson) files."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(annotations_path) as f:
        annotations = ndjson.load(f)
    return metadata, annotations


def write_annotations(output_dir, annotations, coordinate_space, colors):
    """
    Create a neuroglancer annotation folder with the given annotations.
    
    See https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/annotations.md
    """
    names = [a["metadata"]["annotation_object"]["name"] for a in annotations]
    writer = neuroglancer.write_annotations.AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
            neuroglancer.AnnotationPropertySpec(
                id="name",
                type="uint8",
                enum_values=[i for i in range(len(names))],
                enum_labels=names,
            ),
        ],
    )

    for i, a in enumerate(annotations):
        data = a["data"]
        size = a["metadata"]["annotation_object"]["diameter"]
        # TODO not sure what units the diameter is in
        size = size / 1000
        name = i
        color = colors[i]
        for p in data:
            location = [p["location"][k] for k in ["x", "y", "z"]]
            writer.add_point(location, size=size, point_color=color, name=name)

    writer.write(output_dir)


def view_data(coordinate_space, output_dir):
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()

    # Start a static file server, serve the contents of the output directory.
    server = neuroglancer.static_file_server.StaticFileServer(
        static_dir=output_dir,
        bind_address=args.bind_address or "127.0.0.1",
        daemon=True,
    )

    with viewer.txn() as s:
        s.layers["annotations"] = neuroglancer.AnnotationLayer(
            source=f"precomputed://{server.url}",
            tab="rendering",
            shader="""
void main() {
    setColor(prop_point_color());
    setPointMarkerSize(prop_size());
}
    """,
        )
        s.selected_layer.layer = "annotations"
        s.selected_layer.visible = True
        s.show_slices = False

        s.dimensions = coordinate_space

    print(viewer)


def main(paths, output_dir, should_view):
    """For each path set, load the data and write the combined annotations."""
    annotations = []
    for path_set in paths:
        input_metadata_path, input_annotations_path = path_set
        metadata, data = load_data(input_metadata_path, input_annotations_path)
        annotations.append({"metadata": metadata, "data": data})

    coordinate_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units=["", "", ""], scales=[1, 1, 1]
    )
    colors = [(255, 0, 0, 255), (0, 255, 0, 255)]
    write_annotations(output_dir, annotations, coordinate_space, colors)
    print("Wrote annotations to", output_dir)

    if should_view:
        view_data(coordinate_space, output_dir)

if __name__ == "__main__":
    base_dir = Path("/media/starfish/LargeSSD/data/cryoET/data")
    input_metadata_path = base_dir / "sara_goetz-ribosome-1.0.json"
    input_annotations_path = base_dir / "sara_goetz-ribosome-1.0.ndjson"
    paths = [(input_metadata_path, input_annotations_path)]
    input_metadata_path = base_dir / "sara_goetz-fatty_acid_synthase-1.0.json"
    input_annotations_path = base_dir / "sara_goetz-fatty_acid_synthase-1.0.ndjson"
    paths.append((input_metadata_path, input_annotations_path))
    should_view = False
    main(paths, base_dir / "annotations", should_view)
