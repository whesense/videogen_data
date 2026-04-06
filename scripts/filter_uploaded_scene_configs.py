#!/usr/bin/env python3
"""
One ``rclone lsjson`` on the cleanup destination, then remove per-job YAMLs whose
config_id (filename stem) already exists on the remote under cleanup.dest_path.

Uses ``cleanup.remote``, ``cleanup.dest_path``, and ``rclone_config`` from
``scenarios/<scenario>/base.yaml`` (same layout as pipeline ``cleanup`` uploads:
``remote:dest_path/<config_id>/``).

Examples:
  python scripts/filter_uploaded_scene_configs.py --dry-run
  python scripts/filter_uploaded_scene_configs.py --scenario 01_spatad_cycle
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


def top_level_config_ids_on_remote(
    remote: str,
    dest_path: str,
    rclone_config: Path | None,
) -> set[str]:
    """Directory names directly under ``remote:dest_path`` (one rclone call)."""
    entries = rclone_lsjson(remote, dest_path.rstrip("/"), rclone_config)
    names: set[str] = set()
    for e in entries:
        if not e.get("IsDir", False):
            continue
        raw = (e.get("Path", "") or e.get("Name", "")).rstrip("/")
        if not raw:
            continue
        name = raw.split("/")[-1]
        if name:
            names.add(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove scene YAMLs for config_ids already present under cleanup.dest_path on S3"
    )
    parser.add_argument("--scenario", default="01_spatad_cycle")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Default: scenarios/<scenario>/configs",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_base(args.scenario)
    cl = base.get("cleanup") or {}
    remote = cl.get("remote") or base.get("remote")
    dest_path = cl.get("dest_path")
    rc_conf = resolve_rclone_config(base)

    if not remote or not dest_path:
        print(
            "ERROR: base.yaml must set cleanup.remote and cleanup.dest_path (and optionally rclone_config).",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg_dir = args.configs_dir or (SCENARIOS_DIR / args.scenario / "configs")
    cfg_dir = cfg_dir.resolve()
    if not cfg_dir.is_dir():
        print(f"ERROR: configs dir not found: {cfg_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Listing {remote}:{dest_path.rstrip('/')}/ (one rclone lsjson) …")
    try:
        uploaded = top_level_config_ids_on_remote(remote, dest_path, rc_conf)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print(f"  {len(uploaded)} top-level director(ies) on remote")

    removed = 0
    for path in sorted(cfg_dir.glob("*.yaml")):
        stem = path.stem
        if stem not in uploaded:
            continue
        removed += 1
        if args.dry_run:
            print(f"  would remove {path.name} (found on remote)")
        else:
            path.unlink(missing_ok=True)
            print(f"  removed {path.name}")

    print(f"Done. {'Would remove' if args.dry_run else 'Removed'} {removed} config file(s) under {cfg_dir}")


if __name__ == "__main__":
    main()
