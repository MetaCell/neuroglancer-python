# See https://github.com/seung-lab/cloud-volume/issues/412
import json

from cloudvolume import Skeleton
from cloudvolume import CloudVolume
from pathlib import Path
from cryoet_data_portal_neuroglancer.models.json_generator import (
    SegmentPropertyJSONGenerator,
)


def open_swc(swc_path):
    with open(swc_path, "r") as f:
        data = f.read()
    skeleton = Skeleton.from_swc(data)

    return skeleton


def save_swc(skeleton, output_path, id):
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        # "vertex_attributes": [
        #     {"id": "radius", "data_type": "float32", "num_components": 1}
        # ],
        "scales": [
            {
                "chunk_sizes": [
                    [256, 256, 256]
                ],  # information required by neuroglancer but not used
                "key": "data",
                "resolution": [1, 1, 1],
                "size": [1024, 1024, 1024],
            },
        ],
        "segment_properties": "props",
    }

    skeleton.id = id
    cv = CloudVolume(f"file://{str(output_path)}", info=info, compress=False)
    cv.skeleton.meta.info = info
    cv.skeleton.meta.commit_info()
    cv.skeleton.upload(skeleton)


def generate_properties(output_path, ids, labels):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    generator = SegmentPropertyJSONGenerator(
        ids=ids,
        labels=labels,
    )
    json_res = generator.generate_json()
    print(f"Generated properties for {len(ids)} segments.")
    output_file = output_path / "info"
    output_file.write_text(json.dumps(json_res, indent=2))


def main(output_path, input_folder):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    input_folder = Path(input_folder)
    ids = []
    labels = []
    for swc_file in input_folder.glob("*.swc"):
        print(f"Processing {swc_file}")
        skeleton = open_swc(swc_file)
        id_string = swc_file.stem.split("_")[-1]
        id_ = int(id_string)
        rest = swc_file.stem[: -len(id_string) - 1]
        label = rest.replace("[", "")
        label = label.replace("]", "")
        label = label.strip()
        ids.append(id_)
        labels.append(label)
        save_swc(skeleton, output_path, id_)
        print(f"Saved skeleton with ID {id_} to {output_path}")
    generate_properties(output_path / "skeletons" / "props", ids, labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SWC files to CloudVolume format."
    )
    parser.add_argument(
        "output_path", type=str, help="Path to the output CloudVolume directory."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the folder containing SWC files."
    )

    args = parser.parse_args()
    main(args.output_path, args.input_folder)
