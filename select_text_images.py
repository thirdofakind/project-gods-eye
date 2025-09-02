# select_text_images.py
# Requires: pytesseract, opencv-python, tesseract-ocr
# Usage:
#   python select_text_images.py --processed_base ./processed --selected_base ./selected --csv ./selected_images.csv

import argparse
import csv
import sys
from pathlib import Path
import shutil

import cv2
import pytesseract

def has_clear_text(img_path, min_chars=10):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, ""
    text = pytesseract.image_to_string(img)
    return len(text.strip()) >= min_chars, text.strip()

def scan_images(processed_base: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in processed_base.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def main():
    ap = argparse.ArgumentParser(description="Select processed images with clear text using OCR.")
    ap.add_argument("--processed_base", type=Path, default=Path("./processed"),
                    help="Folder with processed images (default: ./processed)")
    ap.add_argument("--selected_base", type=Path, default=Path("./selected"),
                    help="Folder to copy selected images (default: ./selected)")
    ap.add_argument("--csv", type=Path, default=Path("./selected_images.csv"),
                    help="CSV log of selected images (default: ./selected_images.csv)")
    ap.add_argument("--min_chars", type=int, default=10,
                    help="Minimum number of characters to consider text as 'clear' (default: 10)")
    args = ap.parse_args()

    args.selected_base.mkdir(parents=True, exist_ok=True)

    images = scan_images(args.processed_base)
    print(f"[select_text_images] Found {len(images)} images to scan.")

    selected = []
    for i, img_path in enumerate(images, 1):
        ok, text = has_clear_text(img_path, min_chars=args.min_chars)
        if ok:
            # Copy to selected folder, preserving subfolder structure
            rel_path = img_path.relative_to(args.processed_base)
            out_path = args.selected_base / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_path)
            selected.append({"image": str(img_path), "selected_path": str(out_path), "text": text})
        if i % 100 == 0 or i == len(images):
            print(f"\r  scanned {i}/{len(images)} images...", end="", flush=True)

    print(f"\n[select_text_images] Selected {len(selected)} images with clear text.")

    # Write CSV log
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "selected_path", "text"])
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    print(f"[select_text_images] Log written to {args.csv.resolve()}")

if __name__ == "__main__":
    sys.exit(main())