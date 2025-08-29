import re
import itertools


def extract_row_col_from_filename(filename):
    """
    Extract row and column information from filename.
    Assumes filenames contain row/col info in a pattern like 'r{row}c{col}' or '{row}_{col}'.

    Args:
        filename: The filename to parse

    Returns:
        tuple: (row, col) or (None, None) if pattern not found
    """
    filename_str = str(filename)

    # Try pattern like 'r0c1' or 'row0col1'
    match = re.search(r"r(?:ow)?(\d+)c(?:ol)?(\d+)", filename_str, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try pattern like '0_1' or '00_01'
    match = re.search(r"(\d+)_(\d+)", filename_str)
    if match:
        return int(match.group(1)), int(match.group(2))

    raise ValueError(f"Could not extract row/col from filename: {filename_str}")


def compute_grid_dimensions(file_list):
    """
    Compute the number of rows and columns from the available files.

    Args:
        file_list: List of file paths or names

    Returns:
        tuple: (num_rows, num_cols)
    """
    if not file_list:
        raise ValueError("File list is empty, cannot compute grid dimensions.")

    max_row = -1
    max_col = -1
    for filepath in file_list:
        row, col = extract_row_col_from_filename(filepath)
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    return max_row + 1, max_col + 1  # Convert from max index to count


def get_grid_coords(num_chunks_per_dim, in_reverse=False):
    coords = itertools.product(
        range(num_chunks_per_dim[0]),
        range(num_chunks_per_dim[1]),
        range(num_chunks_per_dim[2]),
    )
    if in_reverse:
        iter_coords = list(coords)
        iter_coords.reverse()
    else:
        iter_coords = coords
        return iter_coords
