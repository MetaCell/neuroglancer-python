from cloudvolume import CloudVolume
import numpy as np
from PIL import Image
from pathlib import Path


HOMEPATH = Path.home()
FILEPATH = HOMEPATH / "Documents" / "mc_logo.png"


def create_mc_logo(output_path="file://datasets/mc_logo"):
    img = Image.open(FILEPATH)
    luminance = np.array(img)[:, :, -1]
    Image.fromarray(luminance).save(HOMEPATH / "Documents" / "mc_logo_gray.png")
    imgdata = luminance.T
    dim_size = int((imgdata.shape[0] + imgdata.shape[1]) / 2)
    rawdata = np.zeros(
        shape=(imgdata.shape[0], imgdata.shape[1], dim_size),
        dtype=np.uint8,
    )
    for i in range(dim_size):
        rawdata[:, :, i] = imgdata

    # CloudVolume.from_numpy(
    #     rawdata,
    #     vol_path=output_path,
    #     resolution=(1, 1, 1),
    #     layer_type="image",
    #     progress=True,
    #     compress=False,
    # )
    return rawdata


if __name__ == "__main__":
    create_mc_logo()
