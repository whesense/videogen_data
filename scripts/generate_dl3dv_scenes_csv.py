#!/usr/bin/env python3
"""
Generate DL3DV scenes.csv from S3 zip objects.

Reads keys under:
  s3://our-datasets/videogen_data/04_3dgs_dl3dv/<batch>/<scene>.zip

Writes CSV columns:
  batch,scene,chunk
where:
  chunk = index // chunk_size   (default chunk_size=100)

Examples:
  python scripts/generate_dl3dv_scenes_csv.py
  python scripts/generate_dl3dv_scenes_csv.py --chunk-size 100
  python scripts/generate_dl3dv_scenes_csv.py --exclude-processed --processed-src s3://our-datasets/videogen_data/04_3dgs_dl3dv/
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = "s3://raw-datasets/DL3DV/"
DEFAULT_OUT = REPO_ROOT / "scenarios" / "04_3dgs_dl3dv" / "configs" / "scenes.csv"
DEFAULT_AWS_PROFILE = "s3-av"
AWS_DIR = Path("/home/jovyan/users/shirokov/.aws")
ENV = {
    "AWS_SHARED_CREDENTIALS_FILE": str(AWS_DIR / "credentials"),
    "AWS_CONFIG_FILE": str(AWS_DIR / "config"),
}

def list_s3_keys(src: str, profile: str) -> list[str]:
    cmd = [
        "aws",
        "--profile",
        profile,
        "s3",
        "--endpoint-url",
        "https://s3.cloud.ru",
        "ls",
        "--recursive",
        src.rstrip("/") + "/",
    ]
    env = os.environ.copy()
    env.update(ENV)
    env["AWS_PROFILE"] = profile
    if shutil.which("aws", path=env.get("PATH")) is None:
        raise RuntimeError(
            "aws CLI not found in PATH. Install awscli or add it to PATH, "
            "then rerun this script."
        )
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"aws s3 ls failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr or proc.stdout}"
        )

    keys: list[str] = []
    for line in proc.stdout.splitlines():
        # Typical line: 2026-04-23 11:00:00    123456 batch/scene.zip
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        key = parts[3].strip()
        if key:
            keys.append(key)
    return keys


def extract_batch_scene(keys: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    
    for key in keys:
        if not key.endswith(".zip"):
            continue
        parts = key.split("/")
        if len(parts) != 3:
            # Expect exactly "DL3DV/<batch>/<scene>.zip"
            continue
        batch = parts[-2].strip()
        scene = parts[-1][:-4].strip()  # remove ".zip"
        if not batch or not scene:
            continue
        item = (batch, scene)
        rows.append(item)

    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def extract_processed_pairs(keys: list[str]) -> set[tuple[str, str]]:
    """
    Extract (batch, scene) from processed S3 keys.

    Supports layouts like:
      <batch>/<scene>.zip
      <root>/<batch>/<scene>.zip
      <batch>/<scene>/...
    """
    processed: set[tuple[str, str]] = set()
    for key in keys:
        raw = key.strip().strip("/")
        if not raw:
            continue

        parts = [p for p in raw.split("/") if p]
        if len(parts) < 2:
            continue

        batch = parts[-2].strip()
        scene_part = parts[-1].strip()
        if not batch or not scene_part:
            continue

        if "." in scene_part:
            scene = scene_part.rsplit(".", 1)[0].strip()
        else:
            scene = scene_part

        if scene:
            processed.add((batch, scene))

    return processed


def write_csv(rows: list[tuple[str, str]], output_path: Path, chunk_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "scene", "chunk"])
        for idx, (batch, scene) in enumerate(rows):
            writer.writerow([batch, scene, idx // chunk_size])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenes.csv from DL3DV S3 zip objects")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"S3 prefix (default: {DEFAULT_SRC})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size used for chunk=index//chunk_size")
    parser.add_argument(
        "--profile",
        default=DEFAULT_AWS_PROFILE,
        help=f"AWS profile from ~/.aws/credentials (default: {DEFAULT_AWS_PROFILE})",
    )
    parser.add_argument(
        "--exclude-processed",
        action="store_true",
        help="Exclude scenes that already exist under --processed-src",
    )
    parser.add_argument(
        "--processed-src",
        default="s3://our-datasets/videogen_data/04_3dgs_dl3dv/",
        help="S3 prefix with already processed scenes (used with --exclude-processed)",
    )
    args = parser.parse_args()

    keys = list_s3_keys(args.src, args.profile)
    
    print(f"Found {len(keys)} paths under {args.src}")
    
    rows = extract_batch_scene(keys)

    print(f"Found {len(rows)} scene zip(s) under {args.src}")

    if args.exclude_processed:
        processed_keys = list_s3_keys(args.processed_src, args.profile)
        processed_pairs = extract_processed_pairs(processed_keys)
        before = len(rows)
        rows = [row for row in rows if row not in processed_pairs]
        skipped = before - len(rows)
        print(f"Found {len(processed_pairs)} processed scene(s) under {args.processed_src}")
        print(f"Filtered out {skipped} already processed scene(s)")
    
    write_csv(rows, args.output, args.chunk_size)
    
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
