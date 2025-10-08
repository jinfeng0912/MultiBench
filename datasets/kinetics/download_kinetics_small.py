import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import sys

from .build_kinetics_small_batches import build_batches


# Primary sources (GitHub Raw)
DEFAULT_TRAIN_CSV = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/master/kinetics-400_train.csv"
DEFAULT_VAL_CSV = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/master/kinetics-400_val.csv"
DEFAULT_TEST_CSV = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/master/kinetics-400_test.csv"

# Mirror sources
MIRROR_TRAIN = [
    DEFAULT_TRAIN_CSV,
    "https://cdn.jsdelivr.net/gh/cvdfoundation/kinetics-dataset@master/kinetics-400_train.csv",
    "https://ghproxy.com/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/master/kinetics-400_train.csv",
]
MIRROR_VAL = [
    DEFAULT_VAL_CSV,
    "https://cdn.jsdelivr.net/gh/cvdfoundation/kinetics-dataset@master/kinetics-400_val.csv",
    "https://ghproxy.com/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/master/kinetics-400_val.csv",
]
MIRROR_TEST = [
    DEFAULT_TEST_CSV,
    "https://cdn.jsdelivr.net/gh/cvdfoundation/kinetics-dataset@master/kinetics-400_test.csv",
    "https://ghproxy.com/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/master/kinetics-400_test.csv",
]

# Fallback DeepMind URLs (may be blocked or deprecated in some regions)
FALLBACK_TRAIN_CSV = "https://storage.googleapis.com/deepmind-data/kinetics/annotations/train.csv"
FALLBACK_VAL_CSV = "https://storage.googleapis.com/deepmind-data/kinetics/annotations/val.csv"
FALLBACK_TEST_CSV = "https://storage.googleapis.com/deepmind-data/kinetics/annotations/test.csv"


def download_file(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Try Python requests first (no external deps)
    try:
        import requests  # type: ignore

        with requests.get(url, stream=True, timeout=20) as r:
            if r.status_code != 200:
                return False
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        pass

    # Fallback to curl if available
    try:
        subprocess.run(["curl", "-L", "-sS", "-o", str(out_path), url], check=True)
        return True
    except Exception:
        return False


def load_annotation(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_by_classes(rows: List[Dict[str, str]], classes: Sequence[str]) -> List[Dict[str, str]]:
    allowed = set(classes)
    return [r for r in rows if r.get("label") in allowed or r.get("class") in allowed]


def video_id_to_url(video_id: str, start: str = None, end: str = None) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    if start:
        base += f"&t={start}s"
    return base


def download_split(rows: List[Dict[str, str]], split_dir: Path, per_class_limit: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    # Group by class
    by_class: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        label = r.get("label") or r.get("class") or "unknown"
        by_class.setdefault(label, []).append(r)
    for label, items in by_class.items():
        out_dir = split_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in items[:per_class_limit]:
            vid = r.get("youtube_id") or r.get("youtube-id") or r.get("video_id") or r.get("id")
            if not vid:
                continue
            url = video_id_to_url(vid, r.get("time_start") or r.get("start"), r.get("time_end") or r.get("end"))
            # Use yt-dlp to download best mp4
            cmd = [
                "yt-dlp",
                "-f",
                "best[ext=mp4]/best",
                "-o",
                str(out_dir / f"%(id)s.%(ext)s"),
                url,
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a small Kinetics-400 subset and build MultiBench batches")
    p.add_argument("--raw_root", required=True, help="Where to store raw videos (train/valid/test/<class>)")
    p.add_argument("--out_root", required=True, help="Where to write kinetics_small .pdt/.pkt")
    p.add_argument("--categories", nargs="*", default=["archery", "breakdancing", "crying", "dining", "singing"], help="5-class subset")
    p.add_argument("--per_class_train", type=int, default=60, help="videos per class for train")
    p.add_argument("--per_class_val", type=int, default=10, help="videos per class for valid")
    p.add_argument("--per_class_test", type=int, default=10, help="videos per class for test")
    p.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV, help="train annotation CSV or URL")
    p.add_argument("--val_csv", default=DEFAULT_VAL_CSV, help="val annotation CSV or URL")
    p.add_argument("--test_csv", default=DEFAULT_TEST_CSV, help="test annotation CSV or URL")
    return p.parse_args()


def maybe_download_csv(src: str, dest: Path, fallback: str = None, mirrors: list = None) -> Path:
    if src.startswith("http://") or src.startswith("https://"):
        if dest.exists():
            return dest
        urls = []
        urls.append(src)
        if mirrors:
            for u in mirrors:
                if u not in urls:
                    urls.append(u)
        if fallback:
            urls.append(fallback)
        for url in urls:
            if download_file(url, dest):
                return dest
        raise RuntimeError(
            "Failed to download any CSV. Tried: " + ", ".join(urls) +
            "\nHint: manually place CSVs at " + str(dest.parent) + " and re-run, or pass --train_csv/--val_csv/--test_csv with local paths."
        )
    return Path(src)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_dir = out_root / "_annots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_csv = maybe_download_csv(args.train_csv, cache_dir / "train.csv", FALLBACK_TRAIN_CSV, MIRROR_TRAIN)
    val_csv = maybe_download_csv(args.val_csv, cache_dir / "val.csv", FALLBACK_VAL_CSV, MIRROR_VAL)
    test_csv = maybe_download_csv(args.test_csv, cache_dir / "test.csv", FALLBACK_TEST_CSV, MIRROR_TEST)

    train_rows = filter_by_classes(load_annotation(train_csv), args.categories)
    val_rows = filter_by_classes(load_annotation(val_csv), args.categories)
    test_rows = filter_by_classes(load_annotation(test_csv), args.categories)

    # Download
    download_split(train_rows, raw_root / "train", args.per_class_train)
    download_split(val_rows, raw_root / "valid", args.per_class_val)
    download_split(test_rows, raw_root / "test", args.per_class_test)

    # Build batches
    build_batches(raw_root, out_root, args.categories)


if __name__ == "__main__":
    main()


