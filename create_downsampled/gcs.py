import subprocess
from pathlib import Path

from google.cloud import storage


def list_gcs_files(
    bucket_name, prefix="", file_extension="", gcs_project=None, gcs_local_list=None
):
    """
    List files from a Google Cloud Storage bucket that match the given prefix and extension.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix path within the bucket to filter files
        file_extension: File extension to filter for (e.g., '.zarr')

    Returns:
        List of GCS blob names that match the criteria
    """
    if gcs_local_list and gcs_local_list.exists():
        print(f"Loading file list from local file: {gcs_local_list}")
        with open(gcs_local_list, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Found {len(files)} files in local list")
        return files
    client = storage.Client(project=gcs_project)
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    filtered_files = []

    for blob in blobs:
        if blob.name.endswith(file_extension):
            filtered_files.append(blob.name)

    print(
        f"Found {len(filtered_files)} files matching '{file_extension}' extension in bucket '{bucket_name}' with prefix '{prefix}'"
    )
    return filtered_files


def gcloud_download_dir(gs_prefix: str, local_dir: Path, gcs_project) -> None:
    """
    Recursively download a GCS prefix to a local directory using gcloud.
    Example gs_prefix: 'gs://my-bucket/some/prefix/'
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # Use a list (no shell=True) to avoid injection issues
    cmd = [
        "gcloud",
        "storage",
        "cp",
        "--recursive",
        "--project",
        gcs_project,
        gs_prefix,
        str(local_dir),
    ]

    print("Running command:", " ".join(cmd))
    try:
        res = subprocess.run(
            cmd,
            check=True,  # raises CalledProcessError on nonzero exit
            capture_output=True,  # capture logs; integrate with your logger
            text=True,
        )
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
    except subprocess.CalledProcessError as e:
        # Surface meaningful diagnostics
        print("gcloud cp failed:", e.returncode)
        print(e.stdout)
        print(e.stderr)
        raise


def sync_info_to_gcs_output(
    output_path, gcs_output_path, use_gcs_output, gcs_project, gcs_output_bucket_name
):
    """
    Sync the CloudVolume info file to the GCS output bucket.
    This uploads the info file so the bucket is ready to receive the rest of the data.
    """
    local_info_path = output_path / "info"
    gcs_info_path = gcs_output_path + "info"
    upload_file_to_gcs(
        local_info_path,
        gcs_info_path,
        use_gcs_output,
        gcs_project,
        gcs_output_bucket_name,
    )


def upload_file_to_gcs(
    local_file_path,
    gcs_file_path,
    use_gcs_output,
    gcs_project,
    gcs_output_bucket_name,
    overwrite=True,
):
    """
    Upload a single chunk file to the GCS output bucket.

    Args:
        local_file_path: Path to local chunk file
        gcs_file_path: GCS blob path for the chunk
        overwrite: If False, skip upload if file already exists in GCS

    Returns:
        bool: True if successful, False otherwise
    """
    if not use_gcs_output:
        return True

    try:
        client = storage.Client(project=gcs_project)
        bucket = client.bucket(gcs_output_bucket_name)

        blob = bucket.blob(gcs_file_path)

        # Check if file already exists and overwrite is False
        if not overwrite and blob.exists():
            print(f"File {gcs_file_path} already exists in GCS, skipping upload")
            return True

        blob.upload_from_filename(str(local_file_path))

        return True

    except Exception as e:
        print(f"Error uploading chunk {local_file_path} to GCS: {e}")
        return False


def upload_many_blobs_with_transfer_manager(
    bucket_name,
    filenames,
    source_directory="",
    gcs_output_path="",
    gcs_project="",
    uploaded_files=[],
    failed_files=[],
    workers=8,
):
    """Upload every file in a list to a bucket, concurrently in a process pool.

    Each blob name is derived from the filename, not including the
    `source_directory` parameter. For complete control of the blob name for each
    file (and other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # A list (or other iterable) of filenames to upload.
    # filenames = ["file_1.txt", "file_2.txt"]

    # The directory on your computer that is the root of all of the files in the
    # list of filenames. This string is prepended (with os.path.join()) to each
    # filename to get the full path to the file. Relative paths and absolute
    # paths are both accepted. This string is not included in the name of the
    # uploaded blob; it is only used to find the source files. An empty string
    # means "the current working directory". Note that this parameter allows
    # directory traversal (e.g. "/", "../") and is not intended for unsanitized
    # end user input.
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client(project=gcs_project)
    bucket = storage_client.bucket(bucket_name)

    source_directory = str(source_directory)
    output_filenames = [
        filenames[len(source_directory) + 1 :] for filenames in filenames
    ]

    results = transfer_manager.upload_many_from_filenames(
        bucket,
        output_filenames,
        source_directory=source_directory,
        blob_name_prefix=gcs_output_path,
        max_workers=workers,
        worker_type=transfer_manager.THREAD
    )

    for name, result in zip(filenames, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            failed_files.append(name)
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            uploaded_files.append(name)
