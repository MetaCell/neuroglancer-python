import json
from pathlib import Path

import pandas as pd
from neuroglancer.read_precomputed_annotations import AnnotationReader

FIXED_NGAUTH_START = "gs+ngauth+https://"
NGAUTH_SERVER = ""  # Something like yours.appspot.com
BUCKET_NAME = ""  # Your GCS bucket name
BUCKET_FOLDER = ""  # Folder in your bucket where data is stored
PATH = r"/annotations"

WELL_SIZE = [512, 512]  # Well size in pixels [X, Y]


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


def well_row_col_to_filename(well_row: int, well_col: int) -> str:
    # Ensure well_row and well_col are zero-padded to 2 digits
    # e.g., row 3, col 7 -> r03_c07
    # This matches the folder structure in the GCS bucket

    well_row_str = str(well_row).zfill(2)
    well_col_str = str(well_col).zfill(2)
    return f"{FIXED_NGAUTH_START}{NGAUTH_SERVER}:/{BUCKET_NAME}/{BUCKET_FOLDER}/well_r{well_row_str}_c{well_col_str}"


def main():
    reader = AnnotationReader("file://" + PATH)

    with open(Path(PATH) / "info") as f:
        full_info = json.load(f)
    upper_bound_xy = full_info["upper_bound"][:2][::-1]
    total_well_rows = int(upper_bound_xy[1] // WELL_SIZE[1])
    total_well_cols = int(upper_bound_xy[0] // WELL_SIZE[0])
    print(f"Total well rows: {total_well_rows}, Total well cols: {total_well_cols}")

    annotations = reader.get_within_spatial_bounds()
    annotations = list(annotations)
    print(f"Found {len(annotations)} annotations")

    results = []
    for i, annotation in enumerate(annotations):
        result = map_properties(full_info, annotation, i)
        # Get well row/col by getting s0_start_x and s0_start_y
        start_x = result["s0_start_x"]
        start_y = result["s0_start_y"]
        well_col = int(start_x // WELL_SIZE[0])
        well_row = int(start_y // WELL_SIZE[1])
        filename = well_row_col_to_filename(well_row, well_col)
        result["source_path"] = filename
        # TODO - temp step as only have up to row 10, col 10 right now
        if well_row >= 10 or well_col >= 10:
            continue
        # Create a name from the row + col + field
        row = str(int(result["row"])).zfill(2)
        col = str(int(result["col"])).zfill(2)
        field = str(int(result["field_id"])).zfill(2)
        id = f"{row}_{col}_{field}"
        result["id"] = id
        result["name"] = f"row {row} col {col} field {field}"
        results.append(result)

    df = pd.DataFrame(results)
    # Sort the dataframe by well_row and well_col for easier viewing,
    # so r00_c00 comes first, then r00_c01, etc.
    df = df.sort_values(by=["s0_start_y", "s0_start_x"]).reset_index(drop=True)
    output_path = Path(PATH) / "metadata_filled.csv"
    df.to_csv(output_path, index=False)
    print(f"Wrote annotations to {output_path}")


if __name__ == "__main__":
    main()
