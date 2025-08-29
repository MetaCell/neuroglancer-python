def is_chunk_fully_covered(chunk_bounds, processed_chunks_bounds):
    """
    Check if a chunk is fully covered by processed bounds.

    Args:
        chunk_bounds: [start_coord, end_coord] where each coord is [x, y, z]
        processed_chunks_bounds: List of tuples (start, end) where start and end are [x, y, z]

    Returns:
        bool: True if all 8 corners of the chunk are covered by processed bounds
    """
    if not processed_chunks_bounds:
        return False

    start_coord, end_coord = chunk_bounds
    x0, y0, z0 = start_coord
    x1, y1, z1 = end_coord

    # Generate all 8 corners of the chunk
    corners = [
        [x0, y0, z0],  # min corner
        [x1, y0, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y1, z0],
        [x1, y0, z1],
        [x0, y1, z1],
        [x1, y1, z1],  # max corner
    ]

    # Check if each corner is covered by at least one processed bound
    for corner in corners:
        corner_covered = False
        for start, end in processed_chunks_bounds:
            # Check if corner is inside this processed bound
            if (
                start[0] <= corner[0] < end[0]
                and start[1] <= corner[1] < end[1]
                and start[2] <= corner[2] < end[2]
            ):
                corner_covered = True
                break

        # If any corner is not covered, the chunk is not fully covered
        if not corner_covered:
            return False

    # All corners are covered
    return True