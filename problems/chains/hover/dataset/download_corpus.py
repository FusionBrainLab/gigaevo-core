"""Download and process Wikipedia 2017 abstracts corpus for HoVer retrieval.

Downloads the official HotpotQA Wikipedia abstracts dump and processes it into
a compact JSONL.gz format suitable for BM25 indexing.

Source: https://hotpotqa.github.io/wiki-readme.html
File: enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 (~1.5GB)
License: CC BY-SA 4.0

Usage:
    python -m problems.chains.hover.dataset.download_corpus
"""

import bz2
import gzip
import json
import os
from pathlib import Path
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

                decompressed = bz2.decompress(f.read()).decode("utf-8")

                for line in decompressed.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    title = doc.get("title", "")
                    if isinstance(title, str):
                        title = title.strip()
                    else:
                        title = ""

                    text_field = doc.get("text", [])
                    if isinstance(text_field, list):
                        text = " ".join(str(s).strip() for s in text_field if s).strip()
                    elif isinstance(text_field, str):
                        text = text_field.strip()
                    else:
                        text = ""

                    if not title or not text:
                        continue

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


def build_pkl():
    """Convert .jsonl.gz corpus to .passages.pkl for faster loading."""
    pkl_path = OUTPUT_DIR / "wiki17_abstracts.jsonl.passages.pkl"
    if pkl_path.exists():
        print(f"Pickle already exists: {pkl_path}")
        return

    if not OUTPUT_PATH.exists():
        print(f"Corpus not found at {OUTPUT_PATH}. Run download and process first.")
        return

    import pickle

    print(f"Building pickle from {OUTPUT_PATH}...")
    passages: list[str] = []
    opener = gzip.open if OUTPUT_PATH.suffix == ".gz" else open
    with opener(OUTPUT_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            title = doc.get("title", "")
            text = doc.get("text", "")
            passages.append(f"{title} | {text}")

    with open(pkl_path, "wb") as f:
        pickle.dump(passages, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Saved {len(passages):,} passages to {pkl_path} ({pkl_path.stat().st_size / 1e6:.0f}MB)"
    )


def build_index():
    """Build BM25s index from processed corpus."""
    if BM25S_INDEX_DIR.exists():
        print(f"BM25s index already exists: {BM25S_INDEX_DIR}")
        return

    if not OUTPUT_PATH.exists():
        print(f"Corpus not found at {OUTPUT_PATH}. Run download and process first.")
        return

    from problems.chains.hover.utils.retrieval import build_bm25s_index

    build_bm25s_index(OUTPUT_PATH, BM25S_INDEX_DIR)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_archive()
    process_archive()
    build_pkl()
    build_index()

    if ARCHIVE_PATH.exists() and OUTPUT_PATH.exists():
        size_gb = ARCHIVE_PATH.stat().st_size / (1024**3)
        print(f"\nArchive ({size_gb:.1f}GB) can be removed to save space:")
        print(f"  rm {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
