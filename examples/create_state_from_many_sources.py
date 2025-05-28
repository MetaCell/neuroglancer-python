import numpy as np
from argparse import ArgumentParser

nanometer = 1e-9


def create_gridded_config(
    sources, grid_locations, dim_names, output_dim_names, x_size, y_size, z_size
):
    source_configs = []
    total_n_dims = len(dim_names)
    xyz_index_mapping = [output_dim_names.index(dim) for dim in ["x", "y", "z"]]

    input_dimensions = [[1, ""]] * total_n_dims
    for i, dim_name in enumerate(output_dim_names):
        if dim_name == "x" or dim_name == "y" or dim_name == "z":
            input_dimensions[i] = [nanometer, "m"]
        elif dim_name == "t":
            input_dimensions[i] = [1, "s"]
        elif dim_name == "c'":
            input_dimensions[i] = [1, ""]

    dim_config = {name: val for name, val in zip(output_dim_names, input_dimensions)}
    input_dim_config = {name: val for name, val in zip(dim_names, input_dimensions)}

    for source, grid_location in zip(sources, grid_locations):
        source_config = {
            "url": source,
            "transform": {
                "outputDimensions": dim_config,
                "inputDimensions": input_dim_config,
            },
        }
        source_configs.append(source_config)
        if grid_location == (0, 0, 0):
            continue
        transform_x = grid_location[0] * x_size
        transform_y = grid_location[1] * y_size
        transform_z = grid_location[2] * z_size
        transform_matrix = np.eye(total_n_dims).tolist()
        for i in range(total_n_dims):
            transform_matrix[i].append(0.0)
        transform_matrix[xyz_index_mapping[0]][-1] = transform_x
        transform_matrix[xyz_index_mapping[1]][-1] = transform_y
        transform_matrix[xyz_index_mapping[2]][-1] = transform_z
        source_config["transform"]["matrix"] = transform_matrix

    return source_configs, dim_config


def wrap_config_in_image_layer(source_config, dim_config, final_dim_names):
    combined_config_dim = {k: dim_config[k] for k in final_dim_names if k in dim_config}
    return {
        "dimensions": combined_config_dim,
        "layers": [
            {"type": "image", "source": source_config, "name": "Gridded Image Layer"}
        ],
        "layout": "4panel-alt",
    }


def output_layer_as_json(layer):
    import json

    return json.dumps(layer, indent=4)


def wrap_source_in_ngauth(source_name, ngauth_server, format="zarr3"):
    end = f"|{format}"
    start = "gs+ngauth+https://"
    start += ngauth_server + "/"
    source_name = source_name.replace("gs://", start)
    return source_name + end


def read_sources_from_file(file_path):
    sources = []
    with open(file_path, "r") as file:
        for line in file:
            source = line.strip()
            if source:
                sources.append(source)
    return sources


def parse_grid_locations(sources):
    grid_locations = []
    for source in sources:
        parts = source.strip("/").split("/")
        fname = parts[-1]
        # Format is XYZ_rROWcCOL
        row_id_start = fname.find("r") + 1
        row_id = int(fname[row_id_start : row_id_start + 2])
        col_id_start = fname.find("c") + 1
        col_id = int(fname[col_id_start : col_id_start + 2])
        grid_locations.append((col_id, row_id, 0))  # Assuming z=0 for simplicity
    return grid_locations


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a gridded image layer configuration.")
    parser.add_argument(
        "--ngauth-server",
        "-n",
        type=str,
        help="NGAUTH server to use for authentication.",
        required=True,
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="zarr3",
        help="Format of the source data (default: zarr3).",
    )
    parser.add_argument(
        "--output-dim-names",
        "-o",
        type=str,
        nargs="+",
        default=["z", "t", "c'", "y", "x"],
        help="Names of the dimensions in the data.",
    )
    parser.add_argument(
        "--dim-names",
        "-d",
        type=str,
        nargs="+",
        default=["dim_0", "dim_1", "dim_2", "dim_3", "dim_4"],
        help="Names of the dimensions in the data.",
    )
    parser.add_argument(
        "--final-dim-names",
        "-D",
        type=str,
        nargs="+",
        default=["x", "y", "t", "z"],
    )
    parser.add_argument(
        "--xyz-size",
        "-s",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        help="Size of the x, y, and z dimensions in voxels (default: 1, 1, 1).",
    )
    parser.add_argument(
        "--num_sources",
        "-k",
        type=int,
        default=None,
        help="Number of sources to process (default: None, process all sources).",
    )
    parser.add_argument(
        "source_file",
        type=str,
        help="File containing the list of sources, one per line.",
    )
    args = parser.parse_args()
    ngauth_server = args.ngauth_server
    format_ = args.format
    source_file = args.source_file
    if args.num_sources is not None:
        args.num_sources = int(args.num_sources)
    sources = read_sources_from_file(source_file)
    if args.num_sources is not None:
        sources = sources[: args.num_sources]
    grid_locations = parse_grid_locations(sources)
    sources = [
        wrap_source_in_ngauth(source, ngauth_server, format_) for source in sources
    ]
    output_dim_names = args.output_dim_names
    dim_names = args.dim_names
    final_dim_names = args.final_dim_names
    x_size, y_size, z_size = args.xyz_size

    cfg, dim_cfg = create_gridded_config(
        sources, grid_locations, dim_names, output_dim_names, x_size, y_size, z_size
    )
    image_layer = wrap_config_in_image_layer(cfg, dim_cfg, final_dim_names)
    json_output = output_layer_as_json(image_layer)
    # write the output to a file
    with open("gridded_image_layer.json", "w") as f:
        f.write(json_output)
