# frame_preprocess.py
# Requires: opencv-python
# Usage:
#   python frame_preprocess.py
#   (defaults: --log ./ingest_log.csv --processed_base ./processed --frames_base ./frames)

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import os


# ------------------------------- I/O + CSV --------------------------------- #

FIELDNAMES = [
    # existing fields from video_ingest.py
    "filepath", "filename", "ext", "duration_sec", "fps", "total_frames",
    "processed", "processed_at", "output_dir", "note",
    # new preprocessing fields
    "preprocessed", "preprocessed_at", "preproc_dir", "preproc_note",
]

def load_log(log_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load the ingestion CSV. Ensures new preprocessing columns exist in rows.
    """
    log: Dict[str, Dict[str, str]] = {}
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # backfill missing new fields
                if "preprocessed" not in row:
                    row["preprocessed"] = "false"
                if "preprocessed_at" not in row:
                    row["preprocessed_at"] = ""
                if "preproc_dir" not in row:
                    row["preproc_dir"] = ""
                if "preproc_note" not in row:
                    row["preproc_note"] = ""
                log[row["filepath"]] = row
    return log

def save_log(log_path: Path, log: Dict[str, Dict[str, str]]) -> None:
    """
    Write the (possibly updated) log back to disk with the extended schema.
    """
    tmp = log_path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for k in sorted(log.keys()):
            # ensure all fields present
            row = {name: log[k].get(name, "") for name in FIELDNAMES}
            writer.writerow(row)
    tmp.replace(log_path)


# ---------------------------- Progress Utilities ---------------------------- #

def step(msg: str) -> None:
    print(f"[preproc] {msg}")

def inline(msg: str) -> None:
    print(f"\r{msg}", end="", flush=True)


# --------------------------- Image Processing Ops --------------------------- #

def dehaze_fast_but_simple(bgr):
    """
    Fast 'dehaze-like' enhancement:
      - simple white balance (Gray World)
      - CLAHE on L channel (LAB)
      - light unsharp mask for clarity
    Returns enhanced BGR image.
    """
    # Gray-world white balance (approx)
    result = bgr.astype("float32")
    mean = result.mean(axis=(0, 1), keepdims=True) + 1e-6
    gray_mean = mean.mean(axis=2, keepdims=True)
    gain = gray_mean / mean
    result = (result * gain).clip(0, 255).astype("uint8")

    # CLAHE on L channel
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # Unsharp mask (light)
    blur = cv2.GaussianBlur(result, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharp = cv2.addWeighted(result, 1.25, blur, -0.25, 0)
    return sharp

def to_monochrome(bgr):
    """
    Convert to grayscale and denoise lightly with bilateral filter (edge-preserving).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=7)
    return gray

def bump_contrast_for_ocr(gray):
    """
    Contrast bump tuned for OCR:
      - CLAHE to normalize illumination
      - optional slight thresholding (kept commented; enable if needed)
    Returns enhanced grayscale.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)

    # Optional: adaptive threshold to yield binary images (uncomment if desired)
    # out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, blockSize=35, C=10)
    return out


# ----------------------------- Core Processing ------------------------------ #

@dataclass
class PreprocConfig:
    frames_base: Path     # where raw frames live (from extraction)
    processed_base: Path  # where processed frames will be written

def find_frames_folder(row: Dict[str, str], cfg: PreprocConfig) -> Optional[Path]:
    """
    Locate the folder with raw frames for this video:
      1) Prefer row['output_dir'] if present.
      2) Else search cfg.frames_base for a directory containing the filename stem.
    """
    # 1) use recorded output_dir
    out = row.get("output_dir", "").strip()
    if out:
        p = Path(out)
        if p.exists() and p.is_dir():
            return p

    # 2) fallback search by stem
    stem = Path(row["filename"]).stem.lower()
    best: Optional[Path] = None
    for sub in cfg.frames_base.glob("*"):
        if sub.is_dir() and stem in sub.name.lower():
            best = sub
            break
    return best

def ensure_processed_dir(processed_base: Path, frames_dir: Path) -> Path:
    """
    Build a mirrored output directory path under processed_base using the
    raw frames directory name.
    """
    out_dir = processed_base / frames_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def process_one_frameset(frames_dir: Path, out_dir: Path) -> Tuple[int, int]:
    """
    Process all JPG/PNG frames in frames_dir and write to out_dir.
    Pipeline: dehaze -> monochrome -> contrast bump.
    Returns: (total_found, total_written)
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    total_found = len(files)
    total_written = 0

    for i, f in enumerate(files, 1):
        if i == 1 or (i % 200 == 0):
            inline(f"  processing frames: {i}/{total_found}")
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        step1 = dehaze_fast_but_simple(img)
        step2 = to_monochrome(step1)
        step3 = bump_contrast_for_ocr(step2)

        # Save as PNG to avoid JPEG artifacts in OCR stage
        out_path = out_dir / (f.stem + ".png")
        ok = cv2.imwrite(str(out_path), step3)
        if ok:
            total_written += 1

    inline("  processing frames: done\n")
    return total_found, total_written


# --------------------------------- CLI/Main --------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preprocess extracted frames for OCR (dehaze → mono → contrast)."
    )
    ap.add_argument("--log", type=Path, default=Path("./ingest_log.csv"),
                    help="Path to the ingestion CSV (default: ./ingest_log.csv)")
    ap.add_argument("--frames_base", type=Path, default=Path("./frames"),
                    help="Base folder where raw frames are stored (default: ./frames)")
    ap.add_argument("--processed_base", type=Path, default=Path("./processed"),
                    help="Base folder to write OCR-ready frames (default: ./processed)")
    args = ap.parse_args()

    args.processed_base.mkdir(parents=True, exist_ok=True)

    step(f"Log            : {args.log.resolve()}")
    step(f"Frames (raw)   : {args.frames_base.resolve()}")
    step(f"Frames (output): {args.processed_base.resolve()}")

    log = load_log(args.log)

    # Filter: only videos that have frames extracted.
    rows = [r for r in log.values() if r.get("processed", "false").lower() == "true"]

    if not rows:
        step("No processed videos found in log. Run video_ingest.py first.")
        return 0

    processed_count = 0
    total_sets = len(rows)
    total_frames_in = 0
    total_frames_out = 0

    cfg = PreprocConfig(frames_base=args.frames_base, processed_base=args.processed_base)

    for idx, row in enumerate(rows, 1):
        title = row["filename"]
        if row.get("preprocessed", "false").lower() == "true":
            step(f"[{idx}/{total_sets}] Skip (already preprocessed): {title}")
            continue

        step(f"[{idx}/{total_sets}] Finding frames for: {title}")
        frames_dir = find_frames_folder(row, cfg)
        if not frames_dir:
            step(f"  ⚠ could not locate frames directory for: {title}")
            row["preproc_note"] = "frames_dir_not_found"
            continue

        out_dir = ensure_processed_dir(args.processed_base, frames_dir)
        step(f"  frames dir: {frames_dir.name} → out: {out_dir.name}")

        found, written = process_one_frameset(frames_dir, out_dir)
        total_frames_in += found
        total_frames_out += written

        # Mark preprocessed in CSV
        row["preprocessed"] = "true"
        row["preprocessed_at"] = datetime.now().isoformat(timespec="seconds")
        row["preproc_dir"] = str(out_dir.resolve())
        row["preproc_note"] = f"frames_in={found};frames_out={written}"

        save_log(args.log, log)  # persist row as we go
        processed_count += 1

    step(f"Summary: sets={processed_count}/{total_sets}, frames_in={total_frames_in}, frames_out={total_frames_out}")
    save_log(args.log, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
