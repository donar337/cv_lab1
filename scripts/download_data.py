import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DVM_FRONTVIEW_URL = "https://ndownloader.figshare.com/files/34792480"
DEFAULT_OUTPUT_DIR = "data"
ARCHIVE_NAME = "dvm_front_quality.zip"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def run(output_dir: str | Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    extract_dir = output_dir / "dvm_front"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Dataset already extracted at {extract_dir}. Skipping download.")
        return 0

    if not archive_path.exists():
        print(f"Downloading DVM front-view dataset from Figshare...")
        try:
            download_file(DVM_FRONTVIEW_URL, archive_path)
        except requests.RequestException as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return 1
    else:
        print(f"Archive already exists at {archive_path}")

    print(f"Extracting to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)

    print("Done.")
