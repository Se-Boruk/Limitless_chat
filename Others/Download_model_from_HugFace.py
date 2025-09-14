#!/usr/bin/env python3
# Minimal script: download only matching Hugging Face files (no full snapshot)

from huggingface_hub import HfApi, hf_hub_download
from fnmatch import fnmatch
import os

# ——— User Input ———
SOURCE = "https://huggingface.co/jinaai/jina-embeddings-v3"
DEST_DIR = "jina-embeddings-v3"
REVISION = None  # e.g. "main" or commit SHA
PREFERRED = []  # only files matching these patterns
# —————————————————

# Extract repo_id from URL or accept as-is
if SOURCE.startswith("http://") or SOURCE.startswith("https://"):
    parts = SOURCE.rstrip("/").split("/")
    repo_id = "/".join(parts[-2:])
else:
    repo_id = SOURCE

# List all files in the repo
api = HfApi()
all_files = api.list_repo_files(repo_id=repo_id, revision=REVISION)
print(f"Repo: {repo_id}\nTotal files: {len(all_files)}")

# Match preferred patterns
matched_files = [f for f in all_files for pat in PREFERRED if fnmatch(f, pat)]
if not matched_files:
    print("No preferred files found, downloading everything.")
    matched_files = all_files
else:
    print(f"Matched files: {len(matched_files)}")

# Download matched files
os.makedirs(DEST_DIR, exist_ok=True)
for file in matched_files:
    print(f"→ Downloading: {file}")
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        revision=REVISION,
        local_dir=DEST_DIR,
        local_dir_use_symlinks=False
    )

print(f"\n✅ Done. Downloaded to: {DEST_DIR}")
