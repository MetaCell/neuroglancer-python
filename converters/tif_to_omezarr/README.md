# Configuration

To install with conda (anaconda), execute these commands, replacing yourEnvName with a name for the conda environment you like
`conda env create -n yourEnvName -f anaconda.yml`
`conda activate yourEnvName`

Otherwise check the dependencies in the anaconda.yaml file and install them manually in your system, these dependencies are needed for the tool to operate.

```
positional arguments:
  input_tiff_dir        Directory containing tiff files
  output_zarr_ome_dir   Name of directory that will contain OME/zarr datastore

options:
  -h, --help            show this help message and exit
  --chunk_size CHUNK_SIZE
                        Size of chunk (default: 128)
  --obytes OBYTES       number of bytes per pixel in output (default: 0)
  --nlevels NLEVELS     Number of subdivision levels to create, including level 0 (default: 6)
  --max_gb MAX_GB       Maximum amount of memory (in Gbytes) to use; None means no limit (default: None)
  --zarr_only           Create a simple Zarr data store instead of an OME/Zarr hierarchy (default: False)
  --overwrite           Overwrite the output directory, if it already exists (default: False)
  --num_threads NUM_THREADS
                        Advanced: Number of threads to use for processing. Default is number of CPUs (default: 16)
  --algorithm {mean,gaussian,nearest}
                        Advanced: algorithm used to sub-sample the data (default: mean)
  --ranges RANGES       Advanced: output only a subset of the data. Example (in xyz order):
                        2500:3000,1500:4000,500:600 (default: None)
  --first_new_level FIRST_NEW_LEVEL
                        Advanced: If some subdivision levels already exist, create new levels, starting with this
                        one (default: None)
```

In order for the tool to operate properly, the tif files have to placed in a folder and renamed following the slices order.

So, if there is a single tif as a multipages file the simply rename this 01.tif and operate the script as per instructions.

Otherwise if you have multiple tif files, one per page, then rename them in the correct order so that first page will be 01.tif, second page 02.tif, etc.
