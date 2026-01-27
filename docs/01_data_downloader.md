# 01_data_downloader.py

## Objective
- Fetch the Wikidata constraint-correction corpus in either its light “sample” form or the full Figshare dump and place it under `data/raw/<dataset>`.
- Ensure subsequent pipeline stages (`02_dataframe_builder.py` and beyond) can assume a consistent directory layout and ignore downloaded payloads via `data/.gitignore`.

## Inputs & Outputs
- **Inputs:** CLI flags (`--dataset`, `--out`), remote sources (`GITHUB_SAMPLE_ARCHIVE_URL`, Figshare article `13338743`), and the local `data/` directory for staging.
- **Outputs:** Normalised folder `data/raw/<dataset>/` containing `constraints.tsv` plus the `constraint-corrections/` hierarchy, alongside a refreshed `data/.gitignore`.

## Workflow
1. Parse `--dataset {sample,full}` and the optional `--out` directory.
2. Always create `data/.gitignore` with a catch-all rule plus the exception for the file itself so large artifacts stay untracked.
3. For `sample`, `_github_download_and_extract()` streams the GitHub ZIP, shows progress with `tqdm`, filters the archive down to `constraints.tsv` and `constraint-corrections/`, and re-roots the extracted files so they match the local convention.
4. For `full`, `_figshare_list_files()` queries the Figshare API for article `13338743` and `_figshare_download_file()` streams each asset in chunks, persisting them beside the sample layout.

## Implementation Details
- Both download paths reuse the same `CHUNK_SIZE` and progress instrumentation to make long transfers observable.
- `_github_download_and_extract()` keeps data in-memory via `io.BytesIO` to avoid temporary files, then remaps archive members so constraint folders land inside `data/raw/<dataset>/`.
- Figshare downloads are idempotent: `_figshare_download_file()` skips files already on disk, which lets you resume interrupted runs.
- The module confines all HTTP access to `requests` calls so retry/backoff policies can later be centralized if needed.
