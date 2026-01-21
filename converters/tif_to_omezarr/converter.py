import sys
import re
from pathlib import Path
import json
import shutil
import argparse
import copy
import numpy as np
import tifffile
import zarr
import skimage.transform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import xml.etree.ElementTree as ET


def parseSlices(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print(
            "Could not parse ranges argument '%s'; expected 3 comma-separated ranges"
            % istr
        )
        return None
    slices = []
    for sstr in sstrs:
        if sstr == "":
            slices.append(None)
            continue
        parts = sstr.split(":")
        if len(parts) == 1:
            slices.append(slice(int(parts[0])))
        else:
            iparts = [None if p == "" else int(p) for p in parts]
            if len(iparts) == 2:
                iparts.append(None)
            slices.append(slice(iparts[0], iparts[1], iparts[2]))
    return slices


# return None if succeeds, err string if fails
def create_ome_dir(zarrdir):
    # complain if directory already exists
    if zarrdir.exists():
        err = "Directory %s already exists" % zarrdir
        print(err)
        return err

    try:
        zarrdir.mkdir()
    except Exception as e:
        err = "Error while creating %s: %s" % (zarrdir, e)
        print(err)
        return err


def create_ome_headers(zarrdir, nlevels, has_channels=False):
    axes = []
    if has_channels:
        axes.append({"name": "c", "type": "channel"})
    axes.extend(
        [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
    )

    zattrs_dict = {
        "multiscales": [{"axes": axes, "datasets": [], "name": "/", "version": "0.4"}]
    }

    dataset_dict = {
        "coordinateTransformations": [{"scale": [], "type": "scale"}],
        "path": "",
    }

    zgroup_dict = {"zarr_format": 2}

    datasets = []
    for l in range(nlevels):
        ds = copy.deepcopy(dataset_dict)
        ds["path"] = "%d" % l
        scale = 2.0**l
        if has_channels:
            # Channel scale is always 1 (no downsampling in channel dimension)
            ds["coordinateTransformations"][0]["scale"] = [1.0, scale, scale, scale]
        else:
            ds["coordinateTransformations"][0]["scale"] = [scale] * 3
        # print(json.dumps(ds, indent=4))
        datasets.append(ds)
    zad = copy.deepcopy(zattrs_dict)
    zad["multiscales"][0]["datasets"] = datasets
    json.dump(zgroup_dict, (zarrdir / ".zgroup").open("w"), indent=4)
    json.dump(zad, (zarrdir / ".zattrs").open("w"), indent=4)


def slice_step_is_1(s):
    if s is None:
        return True
    if s.step is None:
        return True
    if s.step == 1:
        return True
    return False


def slice_start(s):
    if s.start is None:
        return 0
    return s.start


def slice_count(s, maxx):
    mn = s.start
    if mn is None:
        mn = 0
    mn = max(0, mn)
    mx = s.stop
    if mx is None:
        mx = maxx
    mx = min(mx, maxx)
    return mx - mn


def detect_ome_planar_channels(tiff_path):
    """Check if TIFF has OME-XML metadata indicating planar multichannel storage.
    Returns (num_channels, num_z_slices) if detected, or (None, None) otherwise."""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            # Try to get OME-XML from ImageDescription
            if (
                hasattr(tif.pages[0], "tags")
                and "ImageDescription" in tif.pages[0].tags
            ):
                desc = tif.pages[0].tags["ImageDescription"].value
                if isinstance(desc, bytes):
                    desc = desc.decode("utf-8", errors="ignore")

                # Check if it contains OME-XML
                if "OME" in desc and "xmlns" in desc:
                    # Parse XML
                    # Remove XML declaration if present
                    if desc.startswith("<?xml"):
                        desc = desc[desc.find("?>") + 2 :].strip()

                    # Extract the OME root element
                    root = ET.fromstring(desc)

                    # Handle namespace
                    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                    if root.tag.startswith("{"):
                        ns_url = root.tag[1 : root.tag.index("}")]
                        ns = {"ome": ns_url}

                    # Find Pixels element
                    pixels = root.find(".//ome:Pixels", ns)
                    if pixels is not None:
                        size_c = pixels.get("SizeC")
                        size_z = pixels.get("SizeZ")
                        num_pages = len(tif.pages)

                        if size_c and size_z:
                            num_c = int(size_c)
                            num_z = int(size_z)

                            # Check if pages match C*Z (planar configuration)
                            if num_pages == num_c * num_z and num_c > 1:
                                print(
                                    f"Detected OME-XML metadata: {num_c} channels × {num_z} z-slices (planar storage)"
                                )
                                return (num_c, num_z)
    except Exception as e:
        print(f"Note: Could not parse OME metadata: {e}")

    return (None, None)


def tifs2zarr(
    tiffdir,
    zarrdir,
    chunk_size,
    obytes=0,
    slices=None,
    maxgb=None,
    channel=None,
    keep_channels=False,
):
    if slices is None:
        xslice = yslice = zslice = None
    else:
        xslice, yslice, zslice = slices
        if not all([slice_step_is_1(s) for s in slices]):
            err = "All slice steps must be 1 in slices"
            print(err)
            return err
    # Note this is a generator, not a list
    tiffs = tiffdir.glob("*.tif")
    rec = re.compile(r"([0-9]+)\.\w+$")
    # rec = re.compile(r'[0-9]+$')
    inttiffs = {}  # Maps starting z-index to (tiff_path, num_pages)
    z_to_file_page = {}  # Maps each z-index to (tiff_path, page_index)
    current_z = 0

    # First pass: collect all tiffs and determine their page counts
    tiff_list = []
    for tiff in tiffs:
        tname = tiff.name
        match = rec.match(tname)
        if match is None:
            continue
        ds = match.group(1)
        itiff = int(ds)
        tiff_list.append((itiff, tiff))

    if len(tiff_list) == 0:
        err = "No tiffs found"
        print(err)
        return err

    # Sort by the numeric ID
    tiff_list.sort(key=lambda x: x[0])

    # Check first TIFF for OME planar multichannel layout
    ome_channels = None
    ome_z_slices = None
    if len(tiff_list) > 0:
        first_tiff_path = tiff_list[0][1]
        ome_channels, ome_z_slices = detect_ome_planar_channels(first_tiff_path)

    # Second pass: open each tiff to count pages and build mapping
    for itiff, tiff_path in tiff_list:
        try:
            with tifffile.TiffFile(tiff_path) as tif:
                num_pages = len(tif.pages)
                print(f"File {tiff_path.name}: {num_pages} page(s)")
                inttiffs[itiff] = (tiff_path, num_pages)

                # If OME planar channels detected and keep_channels is True, use that layout
                if ome_channels is not None and keep_channels:
                    # Pages are organized as: all z-slices for channel 0, then all for channel 1, etc.
                    # We'll handle this in the reading loop
                    for page_idx in range(num_pages):
                        z_to_file_page[current_z] = (tiff_path, page_idx)
                        current_z += 1
                else:
                    # Standard: each page is a z-slice
                    for page_idx in range(num_pages):
                        z_to_file_page[current_z] = (tiff_path, page_idx)
                        current_z += 1
        except Exception as e:
            err = f"Error reading {tiff_path}: {e}"
            print(err)
            return err

    if len(inttiffs) == 0:
        err = "No valid tiffs found"
        print(err)
        return err

    # Get all z-indices
    all_z_indices = sorted(z_to_file_page.keys())
    z0 = 0

    if zslice is not None:
        maxz = all_z_indices[-1] + 1
        valid_zs = range(maxz)[zslice]
        all_z_indices = list(filter(lambda z: z in valid_zs, all_z_indices))
        if zslice.start is None:
            z0 = 0
        else:
            z0 = zslice.start

    if len(all_z_indices) == 0:
        err = "No slices to process after applying z-range filter"
        print(err)
        return err

    minz = all_z_indices[0]
    maxz = all_z_indices[-1]
    cz = maxz - z0 + 1

    # Read first page to get dimensions and dtype
    first_tiff_path, first_page_idx = z_to_file_page[minz]
    try:
        with tifffile.TiffFile(first_tiff_path) as tif:
            tiff0 = tif.pages[first_page_idx].asarray()
    except Exception as e:
        err = "Error reading %s page %d: %s" % (first_tiff_path, first_page_idx, e)
        print(err)
        return err

    # Handle multichannel images
    nchannels = 1
    is_planar_ome = False

    # Check for OME planar multichannel layout
    if ome_channels is not None and ome_channels > 1:
        is_planar_ome = True
        if keep_channels:
            nchannels = ome_channels
            cz = ome_z_slices  # Override z count with actual z-slices
            print(f"Using OME planar layout: {nchannels} channels × {cz} z-slices")
        elif channel is not None and channel >= 0:
            if channel >= ome_channels:
                err = f"Channel {channel} out of range (image has {ome_channels} channels)"
                print(err)
                return err
            print(
                f"Extracting channel {channel} from OME planar layout ({ome_channels} channels)"
            )
            nchannels = 1
            cz = ome_z_slices
        else:
            # Average all channels
            print(f"Converting {ome_channels} planar channels to grayscale")
            nchannels = 1
            cz = ome_z_slices

    # Handle interleaved multichannel (channels in same page)
    if tiff0.ndim == 3 and not is_planar_ome:
        # Image has channels (height, width, channels)
        ny0, nx0, nchannels = tiff0.shape
        if keep_channels:
            print(
                f"Detected {nchannels} interleaved channels, keeping all channels (CZYX format)"
            )
            # Don't modify tiff0, keep all channels
        elif channel is None:
            # Convert to grayscale by averaging channels
            print(
                f"Detected {nchannels} interleaved channels, converting to grayscale (averaging all channels)"
            )
            tiff0 = np.mean(tiff0, axis=2).astype(tiff0.dtype)
            nchannels = 1
        elif channel == -1:
            # Convert to grayscale by averaging
            print(
                f"Detected {nchannels} interleaved channels, converting to grayscale (averaging all channels)"
            )
            tiff0 = np.mean(tiff0, axis=2).astype(tiff0.dtype)
            nchannels = 1
        elif 0 <= channel < nchannels:
            print(f"Detected {nchannels} interleaved channels, using channel {channel}")
            tiff0 = tiff0[:, :, channel]
            nchannels = 1
        else:
            err = f"Channel {channel} out of range (image has {nchannels} channels)"
            print(err)
            return err
    elif tiff0.ndim == 2:
        # Single channel grayscale (or one page of planar multichannel)
        ny0, nx0 = tiff0.shape
        if not is_planar_ome:
            if keep_channels:
                print("Image is single-channel")
            if channel is not None and channel != 0 and not is_planar_ome:
                err = f"Image is single-channel but channel {channel} was requested"
                print(err)
                return err
    else:
        err = f"Unsupported image dimensions: {tiff0.shape}"
        print(err)
        return err
    dt0 = tiff0.dtype
    otype = tiff0.dtype
    divisor = 1
    if obytes == 1 and dt0 == np.uint16:
        print("Converting from uint16 in input to uint8 in output")
        otype = np.uint8
        divisor = 256
    elif obytes != 0 and dt0.itemsize != obytes:
        err = "Cannot perform pixel conversion from %s to %d bytes" % (dt0, obytes)
        print(err)
        return err
    else:
        print("Byte conversion: none")
    print("tiff size", nx0, ny0, "z range", minz, maxz)

    cx = nx0
    cy = ny0
    x0 = 0
    y0 = 0
    if xslice is not None:
        cx = slice_count(xslice, nx0)
        x0 = slice_start(xslice)
    if yslice is not None:
        cy = slice_count(yslice, ny0)
        y0 = slice_start(yslice)
    print("cx,cy,cz", cx, cy, cz)
    print("x0,y0,z0", x0, y0, z0)

    # Create zarr array with appropriate shape
    if keep_channels and nchannels > 1:
        print(f"Creating CZYX array with {nchannels} channels")
        array_shape = (nchannels, cz, cy, cx)
        chunk_shape = (1, chunk_size, chunk_size, chunk_size)
    else:
        array_shape = (cz, cy, cx)
        chunk_shape = (chunk_size, chunk_size, chunk_size)

    tzarr = zarr.open(
        str(zarrdir),
        mode="w",
        zarr_format=2,
        shape=array_shape,
        chunks=chunk_shape,
        dtype=otype,
        write_empty_chunks=False,
        fill_value=0,
        compressor=None,
    )

    # nb of chunks in y direction that fit inside of max_gb
    chy = cy // chunk_size + 1
    if maxgb is not None:
        maxy = int((maxgb * 10**9) / (cx * chunk_size * dt0.itemsize))
        chy = maxy // chunk_size
        chy = max(1, chy)

    # For multichannel, reduce chy to account for multiple channels
    if keep_channels and nchannels > 1:
        # Divide available memory by number of channels
        chy = max(1, chy // nchannels)

    # nb of y chunk groups
    ncgy = cy // (chunk_size * chy) + 1
    print("chy, ncgy", chy, ncgy)

    if keep_channels and nchannels > 1:
        # For multichannel, allocate buffer for one channel at a time to save memory
        buf = np.zeros((chunk_size, min(cy, chy * chunk_size), cx), dtype=dt0)
        # Separate buffer for assembling all channels
        channel_buf = np.zeros(
            (nchannels, chunk_size, min(cy, chy * chunk_size), cx), dtype=dt0
        )
    else:
        buf = np.zeros((chunk_size, min(cy, chy * chunk_size), cx), dtype=dt0)

    for icy in range(ncgy):
        ys = icy * chy * chunk_size
        ye = ys + chy * chunk_size
        ye = min(ye, cy)
        if ye == ys:
            break
        prev_zc = -1

        # Cache for currently open TIFF file
        current_tiff_file = None
        current_tiff_path = None

        # Determine actual z-indices to iterate
        if is_planar_ome and keep_channels and nchannels > 1:
            # For OME planar multichannel: iterate through actual z-slices
            z_indices_to_process = range(cz)
        elif is_planar_ome and (channel is not None or channel == -1):
            # For OME planar single channel extraction: iterate through actual z-slices
            z_indices_to_process = range(cz)
        else:
            z_indices_to_process = [z_idx - z0 for z_idx in all_z_indices]

        for z in z_indices_to_process:
            if is_planar_ome and keep_channels and nchannels > 1:
                # OME planar multichannel: read all channels for this z-slice
                z_idx = z
                try:
                    print(
                        "reading z=%d (all %d channels)     " % (z, nchannels), end="\r"
                    )

                    # Read all channels for this z-slice
                    channel_data = np.zeros((ny0, nx0, nchannels), dtype=dt0)

                    for c in range(nchannels):
                        page_idx = c * ome_z_slices + z
                        tiff_path, _ = z_to_file_page[page_idx]

                        # Open new file if needed
                        if current_tiff_path != tiff_path:
                            if current_tiff_file is not None:
                                current_tiff_file.close()
                            current_tiff_file = tifffile.TiffFile(tiff_path)
                            current_tiff_path = tiff_path

                        channel_data[:, :, c] = current_tiff_file.pages[
                            page_idx
                        ].asarray()

                    tarr = channel_data

                except Exception as e:
                    print(f"\nError reading z={z} channels: {e}")
                    tarr = np.zeros((ny0, nx0, nchannels), dtype=dt0)
            else:
                # Standard processing or single channel from planar
                if is_planar_ome and channel is not None and channel >= 0:
                    # Extract specific channel from planar layout
                    z_idx = z
                    page_idx = channel * ome_z_slices + z
                else:
                    z_idx = all_z_indices[z] if z < len(all_z_indices) else z
                    page_idx = z_to_file_page.get(z_idx, (None, 0))[1]

                tiff_path, page_idx = z_to_file_page.get(
                    page_idx if is_planar_ome else z_idx, (None, 0)
                )

                if tiff_path is None:
                    tarr = np.zeros((ny0, nx0), dtype=dt0)
                    continue

                try:
                    print(
                        "reading z=%d (file: %s, page: %d)     "
                        % (z_idx, tiff_path.name, page_idx),
                        end="\r",
                    )

                    # Open new file if needed
                    if current_tiff_path != tiff_path:
                        if current_tiff_file is not None:
                            current_tiff_file.close()
                        current_tiff_file = tifffile.TiffFile(tiff_path)
                        current_tiff_path = tiff_path

                    tarr = current_tiff_file.pages[page_idx].asarray()

                    # Handle interleaved multichannel images
                    if keep_channels and nchannels > 1 and not is_planar_ome:
                        # Keep all channels - tarr should be (H, W, C)
                        if tarr.ndim != 3 or tarr.shape[2] != nchannels:
                            print(
                                f"\nWarning: Expected shape (H, W, {nchannels}), got {tarr.shape}"
                            )
                            tarr = np.zeros((ny0, nx0, nchannels), dtype=dt0)
                    elif tarr.ndim == 3 and not is_planar_ome:
                        if channel is None or channel == -1:
                            # Convert to grayscale
                            tarr = np.mean(tarr, axis=2).astype(tarr.dtype)
                        else:
                            # Select specific channel
                            tarr = tarr[:, :, channel]
                    elif is_planar_ome and channel == -1:
                        # Average all channels from planar layout
                        channel_sum = tarr.astype(np.float32)
                        for c in range(1, ome_channels):
                            page_idx_c = c * ome_z_slices + z
                            channel_sum += (
                                current_tiff_file.pages[page_idx_c]
                                .asarray()
                                .astype(np.float32)
                            )
                        tarr = (channel_sum / ome_channels).astype(dt0)

                except Exception as e:
                    print("\nError reading", tiff_path, "page", page_idx, ":", e)
                    # If reading fails (file missing or deformed)
                    if keep_channels and nchannels > 1:
                        tarr = np.zeros((ny0, nx0, nchannels), dtype=dt0)
                    else:
                        tarr = np.zeros((ny0, nx0), dtype=dt0)

            if keep_channels and nchannels > 1:
                ny, nx, nc = tarr.shape
                if nx != nx0 or ny != ny0 or nc != nchannels:
                    print(
                        "\nFile %s page %d is the wrong shape (%d, %d, %d); expected %d, %d, %d"
                        % (tiff_path.name, page_idx, nx, ny, nc, nx0, ny0, nchannels)
                    )
                    continue
            else:
                ny, nx = tarr.shape
                if nx != nx0 or ny != ny0:
                    print(
                        "\nFile %s page %d is the wrong shape (%d, %d); expected %d, %d"
                        % (tiff_path.name, page_idx, nx, ny, nx0, ny0)
                    )
                    continue
            if xslice is not None and yslice is not None:
                if keep_channels and nchannels > 1:
                    tarr = tarr[yslice, xslice, :]
                else:
                    tarr = tarr[yslice, xslice]
            cur_zc = z // chunk_size
            if cur_zc != prev_zc:
                if prev_zc >= 0:
                    zs = prev_zc * chunk_size
                    ze = zs + chunk_size
                    if ncgy == 1:
                        print("\nwriting, z range %d,%d" % (zs + z0, ze + z0))
                    else:
                        print(
                            "\nwriting, z range %d,%d  y range %d,%d"
                            % (zs + z0, ze + z0, ys + y0, ye + y0)
                        )
                    if keep_channels and nchannels > 1:
                        tzarr[:, zs:z, ys:ye, :] = channel_buf[
                            :, : ze - zs, : ye - ys, :
                        ]
                        channel_buf[:, :, :, :] = 0
                    else:
                        tzarr[zs:z, ys:ye, :] = buf[: ze - zs, : ye - ys, :]
                        buf[:, :, :] = 0
                prev_zc = cur_zc
            cur_bufz = z - cur_zc * chunk_size
            if keep_channels and nchannels > 1:
                # tarr is (H, W, C), need to transpose to (C, H, W)
                channel_buf[:, cur_bufz, : ye - ys, :] = (
                    np.transpose(tarr[ys:ye, :, :], (2, 0, 1)) // divisor
                )
            else:
                buf[cur_bufz, : ye - ys, :] = tarr[ys:ye, :] // divisor

        # Close any open TIFF file
        if current_tiff_file is not None:
            current_tiff_file.close()

        if prev_zc >= 0:
            zs = prev_zc * chunk_size
            ze = zs + chunk_size
            ze = min(all_z_indices[-1] + 1 - z0 if not is_planar_ome else cz, ze)
            if ze > zs:
                if ncgy == 1:
                    print("\nwriting, z range %d,%d" % (zs + z0, ze + z0))
                else:
                    print(
                        "\nwriting, z range %d,%d  y range %d,%d"
                        % (zs + z0, ze + z0, ys + y0, ye + y0)
                    )
                if keep_channels and nchannels > 1:
                    tzarr[:, zs:ze, ys:ye, :] = (
                        channel_buf[:, : ze - zs, : ye - ys, :] // divisor
                    )
                else:
                    tzarr[zs:ze, ys:ye, :] = buf[: ze - zs, : ye - ys, :] // divisor
            else:
                print("\n(end)")
        if keep_channels and nchannels > 1:
            channel_buf[:] = 0
        else:
            buf[:] = 0

    return nchannels if keep_channels else None


def divp1(s, c):
    n = s // c
    if s % c > 0:
        n += 1
    return n


def process_chunk(args):
    idata, odata, z, y, x, cz, cy, cx, algorithm, has_channels = args

    if has_channels:
        # Handle CZYX format - don't downsample channels
        ibuf = idata[
            :,
            2 * z * cz : (2 * z * cz + 2 * cz),
            2 * y * cy : (2 * y * cy + 2 * cy),
            2 * x * cx : (2 * x * cx + 2 * cx),
        ]
    else:
        ibuf = idata[
            2 * z * cz : (2 * z * cz + 2 * cz),
            2 * y * cy : (2 * y * cy + 2 * cy),
            2 * x * cx : (2 * x * cx + 2 * cx),
        ]

    if np.max(ibuf) == 0:
        return  # Skip if the block is empty to save computation

    # pad ibuf to even in all directions (skip channel dimension)
    ibs = ibuf.shape
    if has_channels:
        nc = ibs[0]
        pad = (0, ibs[1] % 2, ibs[2] % 2, ibs[3] % 2)
        if any(pad[1:]):
            ibuf = np.pad(
                ibuf, ((0, 0), (0, pad[1]), (0, pad[2]), (0, pad[3])), mode="symmetric"
            )
    else:
        pad = (ibs[0] % 2, ibs[1] % 2, ibs[2] % 2)
        if any(pad):
            ibuf = np.pad(
                ibuf, ((0, pad[0]), (0, pad[1]), (0, pad[2])), mode="symmetric"
            )

    # algorithms:
    if algorithm == "nearest":
        if has_channels:
            obuf = ibuf[:, ::2, ::2, ::2]
        else:
            obuf = ibuf[::2, ::2, ::2]
    elif algorithm == "gaussian":
        if has_channels:
            # Process each channel separately
            obuf = np.zeros(
                (nc, ibuf.shape[1] // 2, ibuf.shape[2] // 2, ibuf.shape[3] // 2),
                dtype=ibuf.dtype,
            )
            for c in range(nc):
                obuf[c] = np.round(
                    skimage.transform.rescale(ibuf[c], 0.5, preserve_range=True)
                )
        else:
            obuf = np.round(skimage.transform.rescale(ibuf, 0.5, preserve_range=True))
    elif algorithm == "mean":
        if has_channels:
            # Process each channel separately
            obuf = np.zeros(
                (nc, ibuf.shape[1] // 2, ibuf.shape[2] // 2, ibuf.shape[3] // 2),
                dtype=ibuf.dtype,
            )
            for c in range(nc):
                obuf[c] = np.round(
                    skimage.transform.downscale_local_mean(ibuf[c], (2, 2, 2))
                )
        else:
            obuf = np.round(skimage.transform.downscale_local_mean(ibuf, (2, 2, 2)))
    else:
        raise ValueError(f"algorithm {algorithm} not valid")

    if has_channels:
        odata[
            :, z * cz : (z * cz + cz), y * cy : (y * cy + cy), x * cx : (x * cx + cx)
        ] = np.round(obuf)
    else:
        odata[
            z * cz : (z * cz + cz), y * cy : (y * cy + cy), x * cx : (x * cx + cx)
        ] = np.round(obuf)


def resize(zarrdir, old_level, num_threads, algorithm="mean"):
    idir = zarrdir / ("%d" % old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err

    odir = zarrdir / ("%d" % (old_level + 1))
    idata = zarr.open(idir, mode="r")
    print(
        "Creating level",
        old_level + 1,
        "  input array shape",
        idata.shape,
        " algorithm",
        algorithm,
    )

    # Detect if we have channels (CZYX vs ZYX)
    has_channels = len(idata.shape) == 4

    if has_channels:
        nc, sz, sy, sx = idata.shape
        _, cz, cy, cx = idata.chunks
        odata = zarr.open(
            str(odir),
            mode="w",
            zarr_format=2,
            shape=(nc, divp1(sz, 2), divp1(sy, 2), divp1(sx, 2)),
            chunks=idata.chunks,
            dtype=idata.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
        )
    else:
        sz, sy, sx = idata.shape
        cz, cy, cx = idata.chunks
        odata = zarr.open(
            str(odir),
            mode="w",
            zarr_format=2,
            shape=(divp1(sz, 2), divp1(sy, 2), divp1(sx, 2)),
            chunks=idata.chunks,
            dtype=idata.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,
        )

    # Prepare tasks
    tasks = [
        (idata, odata, z, y, x, cz, cy, cx, algorithm, has_channels)
        for z in range(divp1(sz, 2 * cz))
        for y in range(divp1(sy, 2 * cy))
        for x in range(divp1(sx, 2 * cx))
    ]

    # Use ThreadPoolExecutor to process blocks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk, tasks), total=len(tasks)))

    print("Processing complete")


def main():
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create OME/Zarr data store from a set of TIFF files",
    )
    parser.add_argument("input_tiff_dir", help="Directory containing tiff files")
    parser.add_argument(
        "output_zarr_ome_dir",
        help="Name of directory that will contain OME/zarr datastore",
    )
    parser.add_argument("--chunk_size", type=int, default=128, help="Size of chunk")
    parser.add_argument(
        "--obytes", type=int, default=0, help="number of bytes per pixel in output"
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="For multichannel TIFFs: channel index to extract (0-based), or -1 to convert to grayscale by averaging. If not specified, defaults to grayscale conversion.",
    )
    parser.add_argument(
        "--keep_channels",
        action="store_true",
        help="For multichannel TIFFs: preserve all channels in CZYX format instead of converting to single channel. Overrides --channel option.",
    )
    parser.add_argument(
        "--nlevels",
        type=int,
        default=6,
        help="Number of subdivision levels to create, including level 0",
    )
    parser.add_argument(
        "--max_gb",
        type=float,
        default=None,
        help="Maximum amount of memory (in Gbytes) to use; None means no limit",
    )
    parser.add_argument(
        "--zarr_only",
        action="store_true",
        help="Create a simple Zarr data store instead of an OME/Zarr hierarchy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        # default=False,
        help="Overwrite the output directory, if it already exists",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=cpu_count(),
        help="Advanced: Number of threads to use for processing. Default is number of CPUs",
    )
    parser.add_argument(
        "--algorithm",
        choices=["mean", "gaussian", "nearest"],
        default="mean",
        help="Advanced: algorithm used to sub-sample the data",
    )
    parser.add_argument(
        "--ranges",
        help="Advanced: output only a subset of the data.  Example (in xyz order): 2500:3000,1500:4000,500:600",
    )
    parser.add_argument(
        "--first_new_level",
        type=int,
        default=None,
        help="Advanced: If some subdivision levels already exist, create new levels, starting with this one",
    )

    args = parser.parse_args()

    zarrdir = Path(args.output_zarr_ome_dir)
    if zarrdir.suffix != ".zarr":
        print("Name of ouput zarr directory must end with '.zarr'")
        return 1

    tiffdir = Path(args.input_tiff_dir)
    if not tiffdir.exists() and args.first_new_level is None:
        print("Input TIFF directory", tiffdir, "does not exist")
        return 1

    chunk_size = args.chunk_size
    nlevels = args.nlevels
    maxgb = args.max_gb
    zarr_only = args.zarr_only
    overwrite = args.overwrite
    num_threads = args.num_threads
    algorithm = args.algorithm
    obytes = args.obytes
    channel = args.channel
    keep_channels = args.keep_channels

    if keep_channels and channel is not None:
        print("Warning: --keep_channels overrides --channel option")
        channel = None

    print("overwrite", overwrite)
    first_new_level = args.first_new_level
    if first_new_level is not None and first_new_level < 1:
        print("first_new_level must be at least 1")

    slices = None
    if args.ranges is not None:
        slices = parseSlices(args.ranges)
        if slices is None:
            print("Error parsing ranges argument")
            return 1

    print("slices", slices)

    # even if overwrite flag is False, overwriting is permitted
    # when the user has set first_new_level
    if not overwrite and first_new_level is None:
        if zarrdir.exists():
            print("Error: Directory", zarrdir, "already exists")
            return 1

    if first_new_level is None or zarr_only:
        if zarrdir.exists():
            print("removing", zarrdir)
            shutil.rmtree(zarrdir)

    # tifs2zarr(tiffdir, zarrdir, chunk_size, slices=slices, maxgb=maxgb)

    if zarr_only:
        err = tifs2zarr(
            tiffdir,
            zarrdir,
            chunk_size,
            obytes=obytes,
            slices=slices,
            maxgb=maxgb,
            channel=channel,
            keep_channels=keep_channels,
        )
        if err is not None:
            print("error returned:", err)
            return 1
        return

    if first_new_level is None:
        err = create_ome_dir(zarrdir)
        if err is not None:
            print("error returned:", err)
            return 1

    # Create level 0 first to determine if we have channels
    if first_new_level is None:
        print("Creating level 0")
        result = tifs2zarr(
            tiffdir,
            zarrdir / "0",
            chunk_size,
            obytes=obytes,
            slices=slices,
            maxgb=maxgb,
            channel=channel,
            keep_channels=keep_channels,
        )
        if isinstance(result, str):
            print("error returned:", result)
            return 1
        has_channels = result is not None
    else:
        # Detect if existing level 0 has channels
        level0 = zarr.open(zarrdir / "0", mode="r")
        has_channels = len(level0.shape) == 4

    err = create_ome_headers(zarrdir, nlevels, has_channels=has_channels)
    if err is not None:
        print("error returned:", err)
        return 1

    # for each level (1 and beyond):
    existing_level = 0
    if first_new_level is not None:
        existing_level = first_new_level - 1
    for l in range(existing_level, nlevels - 1):
        err = resize(zarrdir, l, num_threads, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1


if __name__ == "__main__":
    sys.exit(main())
