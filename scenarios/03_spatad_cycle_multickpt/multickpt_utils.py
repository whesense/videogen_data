"""
Helpers for ``03_spatad_cycle_multickpt``: splatad run pruning, remote checkpoint listing,
local cleanup, and rclone command construction.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def resolve_rclone_config_path(repo_root: Path, rclone_config: str | None) -> Path | None:
    if not rclone_config:
        return None
    p = Path(rclone_config).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p if p.is_file() else None


def prune_old_splatad_runs(splatad_root: Path, log) -> None:
    """Keep the newest timestamp directory under ``.../<exp>/splatad/``, remove older runs."""
    if not splatad_root.is_dir():
        log.info(f"Prune splatad: skip (no directory {splatad_root})")
        return
    runs = sorted(
        [p for p in splatad_root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(runs) == 0:
        log.info(f"Prune splatad: {splatad_root} — empty, nothing to do")
        return
    if len(runs) == 1:
        log.info(
            f"Prune splatad: {splatad_root} — single run ({runs[0].name}), nothing to remove"
        )
        return
    log.info(
        f"Prune splatad: {splatad_root} — {len(runs)} runs; "
        f"keeping newest ({runs[0].name}), removing older timestamp dirs"
    )
    for old in runs[1:]:
        log.info(f"  rm -rf {old}")
        shutil.rmtree(old, ignore_errors=True)


def latest_splatad_config_yml(splatad_root: Path) -> Path | None:
    """Newest ``config.yml`` under ``splatad/<timestamp>/`` by directory mtime."""
    if not splatad_root.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for run_dir in splatad_root.iterdir():
        if not run_dir.is_dir():
            continue
        cfg = run_dir / "config.yml"
        if cfg.is_file():
            candidates.append((run_dir.stat().st_mtime, cfg))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def rclone_lsjson(
    remote: str, remote_subpath: str, rclone_conf: Path | None
) -> list[dict[str, Any]]:
    target = f"{remote}:{remote_subpath}".rstrip("/")
    cmd = ["rclone", "lsjson", target]
    if rclone_conf is not None:
        cmd.extend(["--config", str(rclone_conf)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rclone lsjson failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stderr or proc.stdout}"
        )
    out = proc.stdout.strip()
    if not out:
        return []
    return json.loads(out)


def latest_splatad_run_name(
    remote: str, splatad_remote_path: str, rclone_conf: Path | None
) -> str | None:
    """Basename of newest directory under ``.../splatad/`` (by ModTime from lsjson)."""
    entries = rclone_lsjson(remote, splatad_remote_path, rclone_conf)
    dirs = [e for e in entries if e.get("IsDir")]
    if not dirs:
        return None
    dirs.sort(key=lambda e: str(e.get("ModTime", "")), reverse=True)
    name = dirs[0].get("Name") or ""
    if not name and dirs[0].get("Path"):
        name = str(dirs[0]["Path"]).strip("/").split("/")[-1]
    return name.rstrip("/") if name else None


def clear_local_splatad_before_checkpoint_sync(dst_ns: Path, exp_name: str, log) -> None:
    """Remove any existing ``<exp>/splatad/<timestamp>/`` dirs so sync does not keep stale runs."""
    splatad_root = dst_ns / exp_name / "splatad"
    if not splatad_root.is_dir():
        return
    subs = [p for p in splatad_root.iterdir() if p.is_dir()]
    if not subs:
        return
    names = ", ".join(p.name for p in subs)
    log.info(
        f"Removing local {len(subs)} timestamp run dir(s) under {splatad_root} before rclone: {names}"
    )
    shutil.rmtree(splatad_root, ignore_errors=True)


def _posix_path_yaml_block(prefix: str, key: str, path: Path) -> list[str]:
    """YAML lines for ``key: !!python/object/apply:pathlib.PosixPath`` + path segments (NerfStudio format)."""
    parts = path.resolve().parts
    lines = [f"{prefix}{key}: !!python/object/apply:pathlib.PosixPath\n"]
    for p in parts:
        if p == "/":
            lines.append(f"{prefix}- /\n")
        else:
            lines.append(f"{prefix}- {p}\n")
    return lines


def rewrite_checkpoint_config_yaml_paths(
    config_path: Path,
    ns_dir: Path,
    scene_dir_target: Path,
) -> None:
    """Rewrite ``output_dir`` and dataparser ``scene_dir`` PosixPath blocks for this job (scenario 03).

    
    """
    ns_dir = ns_dir.resolve()
    scene_dir_target = scene_dir_target.resolve()
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("output_dir:"):
            out.extend(_posix_path_yaml_block("", "output_dir", ns_dir))
            i += 1
            while i < len(lines) and lines[i].strip().startswith("-"):
                i += 1
            continue
        if "scene_dir:" in line and "pathlib.PosixPath" in line:
            prefix = line[: len(line) - len(line.lstrip())]
            out.extend(_posix_path_yaml_block(prefix, "scene_dir", scene_dir_target))
            i += 1
            while i < len(lines) and lines[i].strip().startswith("-"):
                i += 1
            continue
        out.append(line)
        i += 1
    config_path.write_text("".join(out), encoding="utf-8")


def patch_checkpoint_configs_for_local_data(
    ns_dir: Path,
    config_id: str,
    data_root: Path,
    scene_uuid: str,
    log,
) -> None:
    """Rewrite ``output_dir`` and ``scene_dir`` in latest full/cycle ``config.yml`` for this data root."""
    ns_dir = ns_dir.resolve()
    data_root = data_root.resolve()
    raw_scene = data_root / "raw"
    # Must match ``run_cycle_av2_splatad_multickpt.sh`` (``.../shifted/sensor/train/$SEQ``).
    shifted_scene = data_root / "shifted" / "sensor" / "train" / scene_uuid
    for exp in (f"full_av2_{config_id}", f"cycle_av2_{config_id}"):
        cfg_path = latest_splatad_config_yml(ns_dir / exp / "splatad")
        if cfg_path is None:
            continue
        scene_target = raw_scene if exp.startswith("full_av2") else shifted_scene
        rewrite_checkpoint_config_yaml_paths(cfg_path, ns_dir, scene_target)
        log.info(
            f"Patched checkpoint YAML paths: output_dir={ns_dir}, scene_dir={scene_target} ({cfg_path.name})"
        )


def copy_latest_splatad_checkpoint_run(
    ck_remote: str,
    checkpoints_prefix: str,
    exp_name: str,
    dst_ns: Path,
    flags: list[str],
    rclone_conf: Path | None,
    log,
    rclone_copy: Callable[[str, str, Path, list[str] | None], None],
) -> None:
    """Copy the newest ``<exp>/splatad/<timestamp>/`` from remote (``rclone_copy`` = pipeline hook)."""
    splatad_path = f"{checkpoints_prefix}/{exp_name}/splatad"
    run_name = latest_splatad_run_name(ck_remote, splatad_path, rclone_conf)
    if not run_name:
        raise FileNotFoundError(
            f"No splatad timestamp dirs under {ck_remote}:{splatad_path}"
        )
    src = f"{checkpoints_prefix}/{exp_name}/splatad/{run_name}"
    dst = dst_ns / exp_name / "splatad" / run_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        f"Checkpoints [{exp_name}]: {ck_remote}:{src} -> {dst} (latest run only; "
        "cycle run includes all step-*.ckpt in nerfstudio_models/)"
    )
    rclone_copy(ck_remote, src, dst, flags)


def build_multickpt_render_env(
    ns_dir: Path,
    config_id: str,
    data_root: Path,
    scene_uuid: str,
    log,
) -> dict[str, str]:
    """Patch checkpoint YAML paths, prune splatad dirs, resolve env for multickpt shell."""
    ns_dir.mkdir(parents=True, exist_ok=True)
    patch_checkpoint_configs_for_local_data(ns_dir, config_id, data_root, scene_uuid, log)
    full_splatad = ns_dir / f"full_av2_{config_id}" / "splatad"
    cycle_splatad = ns_dir / f"cycle_av2_{config_id}" / "splatad"

    prune_old_splatad_runs(full_splatad, log)
    prune_old_splatad_runs(cycle_splatad, log)

    step1 = latest_splatad_config_yml(full_splatad)
    if step1 is None:
        raise FileNotFoundError(
            f"No full_av2_{config_id}/splatad/*/config.yml under {ns_dir} "
            "(train locally or restore checkpoints via download.checkpoints_from)."
        )
    step4 = latest_splatad_config_yml(cycle_splatad)
    if step4 is None:
        raise FileNotFoundError(
            f"No cycle_av2_{config_id}/splatad/*/config.yml under {ns_dir}."
        )
    cycle_run_dir = step4.parent
    log.info(f"Using full model config: {step1}")
    log.info(f"Using cycle model run dir: {cycle_run_dir}")
    return {
        "STEP1_CONFIG": str(step1.resolve()),
        "STEP4_CONFIG": str(step4.resolve()),
        "CYCLE_RUN_DIR": str(cycle_run_dir.resolve()),
    }


def _sort_multickpt_step_strs(steps: list[str]) -> list[str]:
    def key(s: str) -> tuple[int, str]:
        return (int(s), s) if s.isdigit() else (0, s)

    return sorted(set(steps), key=key)


def multickpt_step_strs_from_ckpts(cycle_run_dir: Path | None) -> list[str]:
    if cycle_run_dir is None:
        return []
    models = cycle_run_dir / "nerfstudio_models"
    if not models.is_dir():
        return []
    out: list[str] = []
    for c in sorted(models.glob("step-*.ckpt"), key=lambda x: x.name):
        name = c.stem
        if name.startswith("step-"):
            out.append(name[5:])
    return _sort_multickpt_step_strs(out)


def multickpt_step_strs_from_reverse_dirs(scene_root: Path) -> list[str]:
    out: list[str] = []
    for p in scene_root.glob("reverse_shifted_*"):
        if p.is_dir():
            out.append(p.name[len("reverse_shifted_") :])
    return _sort_multickpt_step_strs(out)


def mirror_shifted_cameras_to_rendered(
    shifted_scene: Path, rendered_dir: Path, log
) -> None:
    """Copy ``shifted/.../sensors/cameras/*`` → ``<scene_root>/rendered/`` (replaces previous contents)."""
    cam_src = shifted_scene / "sensors" / "cameras"
    if not cam_src.is_dir():
        log.warning(f"multickpt pairs: skip rendered/ mirror — missing {cam_src}")
        return
    rendered_dir.mkdir(parents=True, exist_ok=True)
    for child in list(rendered_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    for item in cam_src.iterdir():
        dest = rendered_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    log.info(f"multickpt pairs: mirrored cameras → {rendered_dir}")


def _build_multickpt_pairs_tree(
    scene_root: Path,
    seq: str,
    sensor_split: str,
    step_strs: list[str],
    log,
) -> None:
    """``pairs/gt`` comes from ``raw/sensors/cameras/`` (once per frame); ``corrupted_<step>/`` from reverse renders."""
    raw_cam_root = scene_root / "raw" / "sensors" / "cameras"
    pairs_dir = scene_root / "pairs"
    gt_pairs = pairs_dir / "gt"

    if pairs_dir.is_dir():
        shutil.rmtree(pairs_dir)
    gt_pairs.mkdir(parents=True, exist_ok=True)
    gt_done: set[tuple[str, str]] = set()

    for step_str in step_strs:
        reverse_scene = scene_root / f"reverse_shifted_{step_str}" / "sensor" / sensor_split / seq
        corrupt_root = reverse_scene / "sensors" / "cameras"
        corrupted_pairs = pairs_dir / f"corrupted_{step_str}"
        if not corrupt_root.is_dir():
            log.warning(f"multickpt pairs: skip step {step_str} — missing {corrupt_root}")
            continue
        corrupted_pairs.mkdir(parents=True, exist_ok=True)

        for cam_dir in sorted(corrupt_root.iterdir()):
            if not cam_dir.is_dir():
                continue
            cam_name = cam_dir.name
            (corrupted_pairs / cam_name).mkdir(parents=True, exist_ok=True)
            for corrupt in sorted(cam_dir.glob("*.jpg")):
                base = corrupt.name
                if base.endswith("_gt-rgb.jpg") or base.endswith("_depth.jpg"):
                    continue
                key = (cam_name, base)
                if key not in gt_done:
                    gt_src = raw_cam_root / cam_name / base
                    if not gt_src.is_file():
                        log.warning(
                            f"multickpt pairs: no raw GT {raw_cam_root / cam_name / base}"
                        )
                        continue
                    (gt_pairs / cam_name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(gt_src, gt_pairs / cam_name / base)
                    gt_done.add(key)
                shutil.copy2(corrupt, corrupted_pairs / cam_name / base)

    log.info(f"multickpt pairs: wrote {pairs_dir} (gt from raw, corrupted_* per step)")


def build_multickpt_pairs_layout(
    data_root: Path,
    sequence: str,
    sensor_split: str,
    cycle_run_dir: Path | None,
    log,
) -> None:
    """Mirror ``rendered/`` and assemble ``pairs/gt`` + ``pairs/corrupted_<step>/`` under ``data_root``."""
    scene_root = data_root.resolve()
    shifted_scene = scene_root / "shifted" / "sensor" / sensor_split / sequence
    rendered_dir = scene_root / "rendered"

    mirror_shifted_cameras_to_rendered(shifted_scene, rendered_dir, log)

    steps = multickpt_step_strs_from_ckpts(cycle_run_dir)
    if not steps:
        steps = multickpt_step_strs_from_reverse_dirs(scene_root)

    _build_multickpt_pairs_tree(scene_root, sequence, sensor_split, steps, log)


def flatten_shifted_traj_remove_images_subdir(shifted_traj: Path) -> None:
    """Hoist files from ``shifted_traj/<camera>/images/`` to ``shifted_traj/<camera>/`` and drop ``images/``."""
    if not shifted_traj.is_dir():
        return
    for cam_dir in shifted_traj.iterdir():
        if not cam_dir.is_dir():
            continue
        img = cam_dir / "images"
        if not img.is_dir():
            continue
        for f in img.iterdir():
            if f.is_file():
                f.replace(cam_dir / f.name)
        shutil.rmtree(img, ignore_errors=True)


def delete_local_multickpt_artifacts(data_root: Path, log) -> None:
    """Remove raw/shifted/shifted_traj/rendered/pairs/reverse_shifted* and ``logs/`` under ``data_root``."""
    log.info("Deleting local intermediates ...")
    for name in ("raw", "shifted", "shifted_traj", "reverse_shifted", "rendered", "pairs", ".av2_links"):
        d = data_root / name
        if d.is_dir():
            log.info(f"  rm -rf {d}")
            shutil.rmtree(d, ignore_errors=True)
    for rev in data_root.glob("reverse_shifted_*"):
        if rev.is_dir():
            log.info(f"  rm -rf {rev}")
            shutil.rmtree(rev, ignore_errors=True)
    logs_root = data_root / "logs"
    if logs_root.is_dir():
        log.info(f"  rm -rf {logs_root}")
        shutil.rmtree(logs_root, ignore_errors=True)


def rclone_cmd_prefix(repo_root: Path, rclone_config: str | None) -> list[str]:
    cmd = ["rclone"]
    conf = resolve_rclone_config_path(repo_root, rclone_config)
    if conf is not None:
        cmd.extend(["--config", str(conf)])
    return cmd


def rclone_upload_copy_command(
    local_src: Path,
    remote: str,
    remote_dest_path: str,
    flags: list[str] | None,
    repo_root: Path,
    rclone_config: str | None,
) -> list[str]:
    """``rclone copy`` local tree to ``remote:remote_dest_path`` (used in cleanup upload)."""
    cmd = rclone_cmd_prefix(repo_root, rclone_config)
    cmd.extend(["copy", str(local_src), f"{remote}:{remote_dest_path}"])
    if flags:
        cmd.extend(flags)
    return cmd
