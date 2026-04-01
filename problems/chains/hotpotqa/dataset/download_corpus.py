"""Download and process HotpotQA Wikipedia 2017 abstracts corpus.

Downloads the official HotpotQA Wikipedia abstracts dump and processes it into
a compact JSONL.gz format. Index building is handled by separate scripts:
  - build_bm25_index.py   (BM25s index)
  - build_colbert_index.py (ColBERT index)

Source: https://hotpotqa.github.io/wiki-readme.html
File: enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 (~1.5GB)
License: CC BY-SA 4.0

Usage:
    python -m problems.chains.hotpotqa.dataset.download_corpus
"""

import bz2
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

DOWNLOAD_URL = (
    "https://nlp.stanford.edu/projects/hotpotqa/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)

OUTPUT_DIR = Path(__file__).parent
ARCHIVE_PATH = (
    OUTPUT_DIR / "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)
OUTPUT_PATH = OUTPUT_DIR / "wiki17_abstracts.jsonl.gz"
BM25S_INDEX_DIR = OUTPUT_DIR / "bm25s_index"


def _ensure_repo_root_on_path() -> None:
    """Ensure repository root is on sys.path for direct script execution."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "problems").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def download_archive():
    """Download the corpus archive if not already present."""
    if ARCHIVE_PATH.exists():
        print(f"Archive already exists: {ARCHIVE_PATH}")
        return

    print(f"Downloading corpus from {DOWNLOAD_URL}...")
    print("This file is ~1.5GB and may take a while.")
    urllib.request.urlretrieve(DOWNLOAD_URL, ARCHIVE_PATH)
    print(f"Downloaded to {ARCHIVE_PATH}")


def process_archive():
    """Extract and process the archive into JSONL.gz format."""
    if OUTPUT_PATH.exists():
        print(f"Output already exists: {OUTPUT_PATH}")
        return

    print(f"Processing archive: {ARCHIVE_PATH}")
    doc_count = 0

    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as out_f:
        with tarfile.open(ARCHIVE_PATH, "r:bz2") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".bz2"):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                # Each inner file is bz2-compressed JSONL
                decompressed = bz2.decompress(f.read()).decode("utf-8")

                for line in decompressed.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract title and text (text is a list of sentences)
                    title = doc.get("title", "")
                    if isinstance(title, str):
                        title = title.strip()
                    else:
                        title = ""

                    text_field = doc.get("text", [])
                    if isinstance(text_field, list):
                        # Join sentences into a single string
                        text = " ".join(str(s).strip() for s in text_field if s).strip()
                    elif isinstance(text_field, str):
                        text = text_field.strip()
                    else:
                        text = ""

                    if not title or not text:
                        continue

                    # Write compact format
                    out_doc = {
                        "id": doc.get("id", str(doc_count)),
                        "title": title,
                        "text": text,
                    }
                    out_f.write(json.dumps(out_doc, ensure_ascii=False) + "\n")
                    doc_count += 1

                    if doc_count % 500_000 == 0:
                        print(f"  Processed {doc_count:,} documents...")

    print(f"Done! Processed {doc_count:,} documents → {OUTPUT_PATH}")


def build_index():
    """Build BM25s index from processed corpus."""
    if not OUTPUT_PATH.exists():
        print(f"Corpus not found at {OUTPUT_PATH}. Run download and process first.")
        return

    _ensure_repo_root_on_path()
    from problems.chains.hotpotqa.utils.retrieval import (
        build_bm25s_index,
        is_bm25s_index_ready,
    )

    if is_bm25s_index_ready(BM25S_INDEX_DIR):
        print(f"BM25s index already exists: {BM25S_INDEX_DIR}")
        return

    if BM25S_INDEX_DIR.exists():
        print(f"Removing incomplete BM25s index: {BM25S_INDEX_DIR}")
        shutil.rmtree(BM25S_INDEX_DIR)

    shard_size = int(os.environ.get("HOTPOT_BM25_SHARD_SIZE", "250000"))
    dtype = os.environ.get("HOTPOT_BM25_DTYPE", "float32")
    int_dtype = os.environ.get("HOTPOT_BM25_INT_DTYPE", "int32")
    print(
        "Building BM25s index with "
        f"shard_size={shard_size:,}, dtype={dtype}, int_dtype={int_dtype}"
    )

    build_bm25s_index(
        OUTPUT_PATH,
        BM25S_INDEX_DIR,
        shard_size=shard_size,
        dtype=dtype,
        int_dtype=int_dtype,
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_archive()
    process_archive()

    print("\nCorpus ready. Build a retrieval index:")
    print("  BM25:    python dataset/build_bm25_index.py")
    print("  ColBERT: python dataset/build_colbert_index.py")

    # Optional: remove archive to save space
    if ARCHIVE_PATH.exists() and OUTPUT_PATH.exists():
        size_gb = ARCHIVE_PATH.stat().st_size / (1024**3)
        print(f"\nArchive ({size_gb:.1f}GB) can be removed to save space:")
        print(f"  rm {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
