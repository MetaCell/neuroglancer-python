from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    buffer: bytearray
    dimensions: tuple[tuple[int, int, int], tuple[int, int, int]]

    def get_name(self):
        """Return the name of the chunk"""
        z_begin, z_end = self.dimensions[0][0], self.dimensions[1][0]
        y_begin, y_end = self.dimensions[0][1], self.dimensions[1][1]
        x_begin, x_end = self.dimensions[0][2], self.dimensions[1][2]
        return f"{x_begin}-{x_end}_{y_begin}-{y_end}_{z_begin}-{z_end}"

    def write_to_directory(self, directory: Path):
        """Write the chunk to the given directory"""
        directory.mkdir(parents=True, exist_ok=True)
        output_filename = self.get_name()
        output_filepath = directory / output_filename
        with open(output_filepath, "wb") as f:
            f.write(self.buffer)
