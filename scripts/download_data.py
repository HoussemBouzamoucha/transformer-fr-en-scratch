import os
import urllib.request
from pathlib import Path

URLS = {
    "train.fr": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.fr.gz",
    "train.en": "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz",
    "val.fr":   "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.fr.gz",
    "val.en":   "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz",
    "test.fr":  "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.fr.gz",
    "test.en":  "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz",
}

RAW_DIR = Path("data/raw/multi30k")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract():
    import gzip
    import shutil

    for name, url in URLS.items():
        gz_path = RAW_DIR / f"{name}.gz"
        out_path = RAW_DIR / name

        if out_path.exists():
            print(f"Already exists: {out_path}")
            continue

        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, gz_path)

        print(f"Extracting {name}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_path.unlink()
        print(f"Saved: {out_path}")

    print("\nDone. Files in data/raw/multi30k/:")
    for f in sorted(RAW_DIR.iterdir()):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        print(f"  {f.name}: {lines} lines")


if __name__ == "__main__":
    download_and_extract()