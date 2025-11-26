import json
from pathlib import Path

import pandas as pd
from neuroglancer.read_precomputed_annotations import AnnotationReader

PATH = r"/annotations"


def map_properties(full_info: dict, annotation, index: int) -> dict:
    output_dict = {}
    properties = full_info["properties"]

    for i, info in enumerate(properties):
        read_value = annotation.props[i]
        name = info["id"]
        value_to_write = float(read_value)
        if "enum_labels" in info:
            enum_index = info["enum_values"].index(int(read_value))
            enum_label = info["enum_labels"][enum_index]
            value_to_write = enum_label
        output_dict[name] = value_to_write
    return output_dict


def main():
    reader = AnnotationReader("file://" + PATH)

    with open(Path(PATH) / "info") as f:
        full_info = json.load(f)

    annotations = reader.get_within_spatial_bounds()
    annotations = list(annotations)
    print(f"Found {len(annotations)} annotations")

    results = []
    for i, annotation in enumerate(annotations):
        results.append(map_properties(full_info, annotation, i))
    df = pd.DataFrame(results)
    output_path = Path(PATH) / "metadata.csv"
    df.to_csv(output_path, index=False)
    print(f"Wrote annotations to {output_path}")


if __name__ == "__main__":
    main()
