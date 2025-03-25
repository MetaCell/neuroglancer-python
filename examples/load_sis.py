import argparse
import webbrowser
import neuroglancer
import neuroglancer.cli
from pathlib import Path
import dask.array as da
import numpy as np

HERE = Path(__file__).parent
# path to the file
FILEPATH = Path("x.sis")


class OlympusSISFile:
    def __init__(self, path):
        self._path = path
        self._file = open(path, "rb")
        self._get_header_size()
        self._file.seek(self.header_size)

        # TEMP - hardcoded values from metadata
        # Something is a litte wrong though as it's too big
        self.width = 3299
        self.height = 3289
        self.depth = 10  # actual 1025
        self.channels = 1  # actual 9

    def read(self, size=-1):
        return self._file.read(size)

    def seek(self, offset, whence=0):
        if whence == 0:
            return self._file.seek(self.header_size + offset)
        elif whence == 1:
            return self._file.seek(offset, 1)
        elif whence == 2:
            return self._file.seek(offset, 2)

    def tell(self):
        return self._file.tell() - self.header_size

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_header_size(self):
        i = 0
        while True and i < 1000000:
            line = self._file.readline().decode("utf-8")
            i += 1
            if line.startswith("Header"):
                break
        # The format is Header = <size>
        # 49 chars for the bit before Header = <size>
        self.header_size = int(line.split("=")[1].strip()) + 49
        print(f"Header size: {self.header_size}")

    def write_header_to_file(self, path):
        self._file.seek(0)
        with open(path, "wb") as f:
            f.write(self._file.read(self.header_size))
        self._file.seek(self.header_size)

    def to_numpy_memmap(self):
        return np.memmap(
            self._path,
            offset=self.header_size,
            dtype=np.uint16,
            mode="r",
            shape=(self.height, self.width, self.depth, self.channels),
        )


def add_image_layer(state, data, name="image"):
    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "c"], units="nm", scales=[1, 1, 1, 1]
    )
    local_volume = neuroglancer.LocalVolume(data, dimensions)
    state.layers.append(
        name=name,
        layer=neuroglancer.ImageLayer(
            source=local_volume,
            volume_rendering_mode="ON",
            volume_rendering_depth_samples=400,
        ),
        shader="""
#uicontrol invlerp normalized
void main() {
    float val = normalized();
    emitRGBA(vec4(val, val, val, val));
    }
    """,
    )
    state.layout = "3d"


def launch_nglancer():
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    return viewer


def main():
    chunks = (512, 512, 1, 1)
    with OlympusSISFile(str(FILEPATH)) as f:
        f.write_header_to_file("header.txt")
        mmapped = f.to_numpy_memmap()
        # TODO would need to replace with a lazy load from the mmap
        dask_array = da.from_array(mmapped, chunks=chunks)
    viewer = launch_nglancer()
    with viewer.txn() as s:
        add_image_layer(s, dask_array, "image")
    webbrowser.open_new(viewer.get_viewer_url())


if __name__ == "__main__":
    main()
