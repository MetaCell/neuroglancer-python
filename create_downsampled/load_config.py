from pathlib import Path
import os

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    print(
        "Warning: python-dotenv not installed. Install with 'pip install python-dotenv' for .env file support."
    )

HERE = Path(__file__).parent


def parse_bool(value):
    """Parse string boolean values to Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "on")


def load_env_config():
    """
    Load configuration from environment variables.
    First tries to load from .env file if python-dotenv is available,
    then falls back to system environment variables.
    """
    # Load .env file if dotenv is available
    env_file = HERE / ".env"
    if HAS_DOTENV and env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded configuration from {env_file}")
    elif env_file.exists():
        print(
            f"Found {env_file} but python-dotenv not available. Using system environment variables only."
        )
    else:
        print("No .env file found. Using system environment variables only.")

    # Load configuration from environment variables with defaults
    config = {
        # Data source configuration
        "USE_GCS_BUCKET": parse_bool(os.getenv("USE_GCS_BUCKET", "false")),
        "GCS_BUCKET_NAME": os.getenv("GCS_BUCKET_NAME", "your-bucket-name"),
        "GCS_PREFIX": os.getenv("GCS_PREFIX", "path/to/zarr/files/"),
        "GCS_FILE_EXTENSION": os.getenv("GCS_FILE_EXTENSION", ".zarr"),
        "GCS_FILES_LOCAL_LIST": Path(os.getenv("GCS_FILES_LOCAL_LIST", "")),
        "GCS_PROJECT": os.getenv("GCS_PROJECT", None),
        # Output GCS bucket configuration (for uploading results)
        "USE_GCS_OUTPUT": parse_bool(os.getenv("USE_GCS_OUTPUT", "false")),
        "GCS_OUTPUT_BUCKET_NAME": os.getenv(
            "GCS_OUTPUT_BUCKET_NAME", "your-output-bucket-name"
        ),
        "GCS_OUTPUT_PREFIX": os.getenv("GCS_OUTPUT_PREFIX", "processed/"),
        "NUM_UPLOAD_WORKERS": int(os.getenv("NUM_UPLOAD_WORKERS", "4")),
        # Local paths (used when USE_GCS_BUCKET is False)
        "INPUT_PATH": Path(os.getenv("INPUT_PATH", "/temp/in")),
        "OUTPUT_PATH": Path(os.getenv("OUTPUT_PATH", "/temp/out")),
        "DELETE_INPUT": parse_bool(os.getenv("DELETE_INPUT", "false")),
        "DELETE_OUTPUT": parse_bool(os.getenv("DELETE_OUTPUT", "false")),
        # Processing settings
        "OVERWRITE": parse_bool(os.getenv("OVERWRITE", "false")),
        "NUM_MIPS": int(os.getenv("NUM_MIPS", "5")),
        "MIP_CUTOFF": int(os.getenv("MIP_CUTOFF", "0")),
        "CHANNEL_LIMIT": int(os.getenv("CHANNEL_LIMIT", "4")),
        "ALLOW_NON_ALIGNED_WRITE": parse_bool(
            os.getenv("ALLOW_NON_ALIGNED_WRITE", "false")
        ),
        # Process possible comma separated list of integers for manual chunk size
        "MANUAL_CHUNK_SIZE": (
            [int(x) for x in str(os.getenv("MANUAL_CHUNK_SIZE", "None")).split(",")]
            if os.getenv("MANUAL_CHUNK_SIZE", "None").lower() != "none"
            else None
        ),
        "MAX_ITERS": int(os.getenv("MAX_ITERS", "10000")),
    }

    print_config(config)

    return config


def print_config(config):
    """Print the loaded configuration."""
    print("Loaded Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
