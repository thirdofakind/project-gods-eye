# video_ingest.py
# Requires: opencv-python
# Usage:
#   python video_ingest.py
#   (defaults: --input ./video --frames_out ./frames --log ./ingest_log.csv)

import cv2
import csv
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".wmv", ".m4v", ".webm"}

def load_log(log_path: Path) -> dict:
    """Load CSV log into memory keyed by absolute filepath."""
    log = {}
    if log_path.exists():
        with log_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                log[row["filepath"]] = row
    return log

def save_log(log_path: Path, log: dict):
    """Write the in-memory log back to CSV (atomic replace)."""
    tmp = log_path.with_suffix(".tmp")
    fieldnames = [
        "filepath","filename","ext","duration_sec","fps","total_frames",
        "processed","processed_at","output_dir","note"
    ]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(log.keys()):
            writer.writerow(log[key])
    tmp.replace(log_path)

def scan_videos(input_dir: Path) -> list:
    """Recursively collect supported videos under input_dir."""
    vids = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            vids.append(p)
    return vids

def get_video_meta(path: Path):
    """Return (fps, total_frames, duration_sec) or None if cannot open."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return fps, total_frames, duration

def unique_frameset_dir(base_out: Path, video_path: Path, duration_sec: float) -> Path:
    """Build a unique folder name for the frame dump."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = video_path.stem[:64]
    dur = f"{int(round(duration_sec))}s"
    return base_out / f"{ts}_{stem}_dur{dur}"

def _print_step(msg: str):
    """Simple step logger."""
    print(f"[video_ingest] {msg}")

def write_frames(video_path: Path, out_dir: Path, frame_stride: int = 1, *,
                 total_frames: int | None = None, verbose: bool = True) -> int:
    """Extract frames to out_dir with a lightweight console progress indicator.

    Args:
        video_path: Source video file.
        out_dir: Directory to write frames.
        frame_stride: Save every Nth frame (1 = every frame).
        total_frames: Optional known frame count to compute %.
        verbose: If True, prints progress to stdout.

    Returns:
        Number of frames written, or -1 if the video couldn't be opened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1

    written = 0
    idx = 0
    last_pct = -1

    ok, frame = cap.read()
    while ok:
        if frame_stride <= 1 or (idx % frame_stride == 0):
            fname = f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            written += 1
        # progress update every ~1%% or every 250 frames if total is unknown
        if verbose:
            if total_frames and total_frames > 0:
                pct = int((idx + 1) * 100 / total_frames)
                if pct != last_pct:
                    print(f"\r  extracting: {pct:3d}% ({idx+1} / {total_frames})", end="", flush=True)
                    last_pct = pct
            elif (idx % 250) == 0:
                print(f"\r  extracting: {idx+1} frames...", end="", flush=True)
        idx += 1
        ok, frame = cap.read()

    cap.release()
    if verbose:
        print("\r  extracting: done" + " " * 20)
    return written

def ensure_row(log: dict, video_path: Path, meta) -> dict:
    """Ensure a log row exists for video_path; create if missing."""
    key = str(video_path.resolve())
    row = log.get(key)
    if row is None:
        fps, total_frames, duration = meta if meta else (0.0, 0, 0.0)
        row = {
            "filepath": key,
            "filename": video_path.name,
            "ext": video_path.suffix.lower(),
            "duration_sec": f"{duration:.3f}",
            "fps": f"{fps:.3f}",
            "total_frames": str(total_frames),
            "processed": "false",
            "processed_at": "",
            "output_dir": "",
            "note": "",
        }
        log[key] = row
    return row

def process_video(video_path: Path, frames_base: Path, log: dict, frame_stride: int):
    """Validate, extract frames, and update the log for one video with progress."""
    _print_step(f"Preparing: {video_path.name}")
    meta = get_video_meta(video_path)
    row = ensure_row(log, video_path, meta)

    if row["processed"].lower() == "true":
        _print_step(f"Skip (already processed): {video_path.name}")
        return

    if video_path.suffix.lower() not in SUPPORTED_EXTS:
        row["note"] = "unsupported_format"
        _print_step(f"Unsupported format: {video_path.name}")
        return

    if not meta:
        row["note"] = "cannot_open_video"
        _print_step(f"Cannot open: {video_path.name}")
        return

    fps, total_frames, duration = meta
    out_dir = unique_frameset_dir(frames_base, video_path, duration)

    _print_step(
        f"Extracting frames → {out_dir.name} | fps={fps:.2f} | frames={total_frames} | dur={duration:.1f}s"
    )

    count = write_frames(
        video_path, out_dir, frame_stride=frame_stride, total_frames=total_frames, verbose=True
    )

    if count < 0:
        row["note"] = "frame_extraction_failed"
        _print_step(f"Extraction failed: {video_path.name}")
        return

    row.update({
        "duration_sec": f"{duration:.3f}",
        "fps": f"{fps:.3f}",
        "total_frames": str(total_frames),
        "processed": "true",
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir.resolve()),
        "note": f"frames_saved={count};stride={frame_stride}",
    })
    _print_step(f"Done: {video_path.name} (saved {count} frames)")

def main():
    ap = argparse.ArgumentParser(
        description="Extract frames from videos and maintain a CSV ingestion log."
    )
    ap.add_argument("--input", type=Path, default=Path("./video"),
                    help="Folder with videos (default: ./video)")
    ap.add_argument("--frames_out", type=Path, default=Path("./frames"),
                    help="Base folder for extracted frames (default: ./frames)")
    ap.add_argument("--log", type=Path, default=Path("./ingest_log.csv"),
                    help="CSV log path (default: ./ingest_log.csv)")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Save every Nth frame (default 1 = every frame)")
    args = ap.parse_args()

    _print_step(f"Input folder: {args.input.resolve()}")
    _print_step(f"Frames out  : {args.frames_out.resolve()}")
    _print_step(f"Log path    : {args.log.resolve()}")

    args.frames_out.mkdir(parents=True, exist_ok=True)
    args.input.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    _print_step("Scanning for videos…")
    log = load_log(args.log)
    videos = scan_videos(args.input)
    _print_step(f"Found {len(videos)} video(s)")

    for v in videos:
        meta = get_video_meta(v)
        ensure_row(log, v, meta)
    save_log(args.log, log)

    processed = 0
    saved_frames = 0

    for i, v in enumerate(videos, start=1):
        _print_step(f"[{i}/{len(videos)}] Processing {v.name}")
        before = int(log[str(v.resolve())]["total_frames"]) if str(v.resolve()) in log and log[str(v.resolve())]["total_frames"].isdigit() else 0
        process_video(v, args.frames_out, log, args.frame_stride)
        save_log(args.log, log)
        if log[str(v.resolve())]["processed"].lower() == "true":
            processed += 1
            note = log[str(v.resolve())]["note"]
            # parse frames_saved from note if present
            try:
                part = next((p for p in note.split(";") if p.startswith("frames_saved=")), None)
                if part:
                    saved_frames += int(part.split("=", 1)[1])
            except Exception:
                pass
        time.sleep(0.01)

    _print_step(f"Summary: processed {processed}/{len(videos)} video(s), saved ~{saved_frames} frame(s)")

if __name__ == "__main__":
    sys.exit(main())