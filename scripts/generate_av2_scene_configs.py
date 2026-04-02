#!/usr/bin/env python3
"""
One-shot: list Argoverse2 train + val scene folders via rclone and write per-scene
YAMLs under scenarios/<scenario>/configs/ (for ray_run.py / run.py).

Reads download.remote, download.src_path, rclone_config from scenarios/<scenario>/base.yaml.

Examples:
  python scripts/generate_av2_scene_configs.py --dry-run
  python scripts/generate_av2_scene_configs.py --force
  python scripts/generate_av2_scene_configs.py --scenario 01_spatad_cycle --splits train val
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = REPO_ROOT / "scenarios"


def load_base(scenario: str) -> dict[str, Any]:
    p = SCENARIOS_DIR / scenario / "base.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def resolve_rclone_config(base: dict[str, Any]) -> Path | None:
    rc = base.get("rclone_config")
    if not rc:
        return None
    p = Path(rc).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p if p.is_file() else None


def rclone_lsjson(remote: str, remote_subpath: str, rclone_config: Path | None) -> list[dict[str, Any]]:
    target = f"{remote}:{remote_subpath}".rstrip("/")
    cmd = ["rclone", "lsjson", target]
    if rclone_config is not None:
        cmd.extend(["--config", str(rclone_config)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rclone lsjson failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr or proc.stdout}"
        )
    out = proc.stdout.strip()
    if not out:
        return []
    return json.loads(out)


def list_scene_ids(
    remote: str,
    src_path: str,
    split: str,
    rclone_config: Path | None,
) -> list[str]:
    path = f"{src_path.rstrip('/')}/{split}"
    entries = rclone_lsjson(remote, path, rclone_config)
    names: list[str] = []
    for e in entries:
        if not e.get("IsDir", False):
            continue
        name = (e.get("Path", "") or e.get("Name", "")).rstrip("/")
        if name:
            names.append(name)
    return sorted(names)


def unique_stem(scene_id: str, occupied: set[str]) -> str:
    stem = scene_id.split("-")[0] if "-" in scene_id else scene_id[:8]
    cand = f"scene_{stem}"
    if cand not in occupied:
        occupied.add(cand)
        return cand
    cand = "scene_" + scene_id.replace("-", "_")
    if cand not in occupied:
        occupied.add(cand)
        return cand
    n = 2
    while True:
        alt = f"scene_{stem}_{n}"
        if alt not in occupied:
            occupied.add(alt)
            return alt
        n += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-scene YAMLs from rclone AV2 listing")
    parser.add_argument("--scenario", default="01_spatad_cycle")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: scenarios/<scenario>/configs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing scene_*.yaml")
    args = parser.parse_args()

    base = load_base(args.scenario)
    dl = base.get("download") or {}
    remote = dl.get("remote") or base.get("remote")
    src_path = dl.get("src_path")
    rc_conf = resolve_rclone_config(base)

    if not remote or not src_path:
        print("ERROR: base.yaml must set download.remote and download.src_path.", file=sys.stderr)
        sys.exit(1)
    if rc_conf is None and base.get("rclone_config"):
        print(f"WARNING: rclone config not found, using default rclone paths", file=sys.stderr)

    out_dir = args.output_dir or (SCENARIOS_DIR / args.scenario / "configs")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stems allocated in this run only (do not pre-seed from disk — would force wrong names)
    occupied: set[str] = set()
    written = 0
    for split in args.splits:
        print(f"Listing {remote}:{src_path.rstrip('/')}/{split}/ ...")
        try:
            scenes = list_scene_ids(remote, src_path, split, rc_conf)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(f"  {len(scenes)} scenes")
        for scene_id in scenes:
            stem = unique_stem(scene_id, occupied)
            path = out_dir / f"{stem}.yaml"
            body = {"download": {"split": split, "scene": scene_id}}
            if path.exists() and not args.force:
                print(f"  skip exists: {path.name}")
                continue
            written += 1
            if args.dry_run:
                print(f"  would write {path.name}")
            else:
                path.write_text(yaml.dump(body, default_flow_style=False, sort_keys=False))
                print(f"  wrote {path.name}")

    print(f"Done. {'Would write' if args.dry_run else 'Wrote'} {written} file(s) under {out_dir}")


if __name__ == "__main__":
    main()
