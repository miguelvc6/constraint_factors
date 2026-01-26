#!/usr/bin/env python3
"""
01_data_downloader.py
====================
Download the BASS Wikidata constraint-correction dataset (sample or full).

Datasets
--------
* **sample** - the sample corpus shipped in the GitHub repository
  `Tpt/bass-materials` -> 1'80GB
* **full**   - the full dump hosted on Figshare
  (article ID 13338743) -> 10'50GB

Usage
-----
python src/01_data_downloader.py --dataset {sample,full}

"""

import argparse
import io
import pathlib
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

FIGSHARE_ID_FULL = 13338743  # “Wikidata Constraint Violations - July 2018 - extended”
GITHUB_SAMPLE_ARCHIVE_URL = "https://github.com/Tpt/bass-materials/archive/refs/heads/master.zip"

# Root directory created by GitHub inside the ZIP (e.g. bass-materials-master/)
ARCHIVE_ROOT_PREFIX = "bass-materials-master/"

# Paths in the archive that we actually want to extract for the sample dataset
ARCHIVE_MEMBERS = [
    f"{ARCHIVE_ROOT_PREFIX}constraints.tsv",
    f"{ARCHIVE_ROOT_PREFIX}constraint-corrections/",
]

CHUNK_SIZE = 1024 * 8  # 8 KiB stream buffer

# ---------------------------------------------------------------------------
# Figshare functions (for the *full* dataset)
# ---------------------------------------------------------------------------


def _figshare_list_files(article_id: int) -> list[dict]:
    """Return metadata for all files attached to a public Figshare article.

    The plain article endpoint only returns the first page of files, so we call
    the dedicated /files endpoint and paginate until we have everything.
    """

    files: list[dict] = []
    offset = 0
    limit = 100
    endpoint = f"https://api.figshare.com/v2/articles/{article_id}/files"

    while True:
        resp = requests.get(endpoint, params={"offset": offset, "limit": limit})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        files.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    return files


def _figshare_download_file(meta: dict, dest_dir: pathlib.Path) -> None:
    """Stream-download a file from Figshare with a progress bar."""
    url = meta["download_url"]
    fname = dest_dir / meta["name"]

    if fname.exists():
        print(f"[skip] {fname} already present")
        return

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(fname, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=fname.name) as bar,
        ):
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


# ---------------------------------------------------------------------------
# GitHub functions (for the *sample* dataset)
# ---------------------------------------------------------------------------


def _github_download_and_extract(dest_dir: pathlib.Path) -> None:
    """Download the GitHub ZIP archive and extract the relevant dataset files."""

    # Fetch archive size first for progress feedback
    head = requests.head(GITHUB_SAMPLE_ARCHIVE_URL)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))

    print("Downloading GitHub sample archive …")
    buffer = io.BytesIO()
    with (
        requests.get(GITHUB_SAMPLE_ARCHIVE_URL, stream=True) as r,
        tqdm(total=total, unit="B", unit_scale=True, desc="bass-materials.zip") as bar,
    ):
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
                bar.update(len(chunk))

    buffer.seek(0)

    print("Extracting dataset files …")
    with zipfile.ZipFile(buffer) as zf:
        members = [m for m in zf.namelist() if any(m.startswith(p) for p in ARCHIVE_MEMBERS)]
        if not members:
            raise RuntimeError("Expected dataset files not found in GitHub archive.")
        for member in members:
            # Compute the relative path inside dest_dir.
            if member.startswith(f"{ARCHIVE_ROOT_PREFIX}constraint-corrections/"):
                rel_path = pathlib.Path(member).relative_to(f"{ARCHIVE_ROOT_PREFIX}constraint-corrections")
            else:
                rel_path = pathlib.Path(member).relative_to(ARCHIVE_ROOT_PREFIX)
            target_path = dest_dir / rel_path
            if member.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
    print("Sample dataset extracted successfully.")


# ---------------------------------------------------------------------------
# Main control flow
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the BASS Wikidata constraint-correction dataset")
    parser.add_argument(
        "--dataset",
        choices=["sample", "full"],
        required=True,
        help="Which dataset to fetch.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("data/raw"),
        help="Output directory for the downloaded dataset files.",
    )
    args = parser.parse_args()

    # Make data folder
    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # .gitignore the data files
    gitignore = Path("data/.gitignore")
    gitignore.write_text("*\n!.gitignore\n")

    if args.dataset == "sample":
        _github_download_and_extract(out_dir)
    else:  # full dataset via Figshare
        print(f"Retrieving file list for Figshare article {FIGSHARE_ID_FULL} …")
        for meta in _figshare_list_files(FIGSHARE_ID_FULL):
            _figshare_download_file(meta, out_dir)
        print("Full dataset downloaded successfully.")


if __name__ == "__main__":
    main()
