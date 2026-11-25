"""Download the Yambda-50M files we need into data/raw."""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "yandex/yambda"
REVISION = "main"  # later: replace with a commit hash to freeze the data version
FILES = [
    "flat/50m/multi_event.parquet",
    "artist_item_mapping.parquet",
    "album_item_mapping.parquet",
]
RAW_DIR = Path("data/raw")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=REVISION,
            local_dir=RAW_DIR,
        )
        print("saved:", path)


if __name__ == "__main__":
    main()