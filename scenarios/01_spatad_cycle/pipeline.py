"""
Pipeline for scenario 01_spatad_cycle.

GT data: Argoverse2 driving dataset scenes (downloaded via rclone)
Cycle output: reverse-shifted views after SPATAD cycle.

Invariant: download -> render -> save_pairs -> cleanup
  render      – trains original + cycle SplatAD, renders shifted & reverse-shifted
  save_pairs  – validates pairs/{gt,corrupted} + comparison video
                (gt = first-shift ``gt-rgb`` renders, not raw camera JPEGs)
  cleanup     – uploads pairs & checkpoints to S3, deletes local intermediates
"""
from __future__ import annotations

import shutil
from pathlib import Path

from lib.base_pipeline import BasePipeline


class Pipeline(BasePipeline):

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def _cycle_dir(self) -> Path:
        return self.config.repo_root / self.param(
            "render", "cycle_recon_dir", default="cycle_reconstruction"
        )

    @property
    def _scene(self) -> str:
        scene = self.param("download", "scene")
        if not scene:
            raise ValueError("download.scene must be set in the job config")
        return scene

    def _download_remote(self) -> str | None:
        """Rclone remote for dataset download (fallback: top-level ``remote``)."""
        return self.param("download", "remote", default=None) or self.param("remote", default=None)

    def _cleanup_remote(self) -> str | None:
        """Rclone remote for upload in cleanup (fallback: top-level ``remote``)."""
        return self.param("cleanup", "remote", default=None) or self.param("remote", default=None)

    def _env(self) -> dict[str, str]:
        """Environment variables passed to run_cycle_av2.sh."""
        return {
            "SEQ": self._scene,
            "CONFIG_ID": self.config.config_id,
            "REPO_DATA": str(self.config.data_root.parent),
            "SHIFT": self.param("render", "shift", default="-3.0 0.0 0.0"),
            "GPU": str(self.param("render", "gpu", default="0")),
            "SPLATAD_NUM_ITER": str(self.param("render", "splatad_num_iter", default=30001)),
        }

    # ── steps ────────────────────────────────────────────────────────────────

    def download(self) -> None:
        """Download a single Argoverse2 scene from S3."""
        remote = self._download_remote()
        if not remote:
            raise ValueError("Set download.remote (or legacy top-level remote) for rclone source")
        src_path = self.param("download", "src_path")
        scene = self._scene
        flags = self.param("download", "flags", default=[])

        full_path = f"{src_path}/{scene}"
        self.log.info(f"Scene  : {scene}")
        self.log.info(f"Source : {remote}:{full_path}")
        self.log.info(f"Dest   : {self.config.raw_dir}")

        self.rclone_copy(remote, full_path, self.config.raw_dir, flags)

    def render(self) -> None:
        """Run full SPATAD cycle reconstruction (Steps 1-5 of the shell script).

        Trains original SplatAD, renders shifted scene, trains cycle model,
        renders reverse-shifted scene.  All nerfstudio outputs land under
        data_root/logs/nerfstudio/ (isolated per config_id for Ray safety).
        """
        cycle_dir = self._cycle_dir
        self.log.info(f"Cycle dir : {cycle_dir}")
        self.log.info(f"Scene     : {self._scene}")

        self.sh(
            "bash run_cycle_av2.sh",
            cwd=cycle_dir,
            env_extra=self._env(),
        )

    def save_pairs(self) -> None:
        """Validate pairs/{gt,corrupted} and comparison videos.

        ``gt`` is taken from the shifted scene tree: prefer
        ``shifted/.../sensors/cameras/<cam>/<timestamp>_gt-rgb.jpg``, else ``shifted/.../gt-rgb/<cam>/<timestamp>.jpg``.
        ``corrupted`` matches ``reverse_shifted/.../sensors/cameras/`` RGB. Steps 6–7 in run_cycle_av2.sh.
        """
        pairs = self.config.pairs_dir
        expected = ("gt", "corrupted")
        missing = [d for d in expected if not (pairs / d).is_dir()]
        if missing:
            self.log.warning(f"Missing pair dirs (render may not have run): {missing}")
        else:
            for d in expected:
                n = sum(1 for _ in (pairs / d).rglob("*.jpg"))
                self.log.info(f"  {d}: {n} images")

        videos = list(self.config.data_root.glob("comparison_*.mp4"))
        self.log.info(f"  comparison videos: {len(videos)}")

    def cleanup(self) -> None:
        """Upload results to S3 and delete local copies of uploaded data.

        Uploads: pairs/, comparison videos, checkpoints (optional).
        After upload, removes raw/, shifted/, reverse_shifted/, rendered/, pairs/,
        and logs/ (nerfstudio). Keeps ``comparison_*.mp4`` in the scene root only.
        """
        remote = self._cleanup_remote()
        dest_path = self.param("cleanup", "dest_path")
        flags = self.param("cleanup", "flags", default=[])
        upload_ckpts = self.param("cleanup", "upload_checkpoints", default=True)
        delete_local = self.param("cleanup", "delete_after_upload", default=True)

        if not remote or not dest_path:
            self.log.warning(
                "cleanup: cleanup.remote (or top-level remote) / dest_path not configured, skipping upload"
            )
            return

        config_id = self.config.config_id
        remote_base = f"{remote}:{dest_path}/{config_id}"

        # 1) Upload pairs
        pairs_dir = self.config.pairs_dir
        if pairs_dir.is_dir() and any(pairs_dir.iterdir()):
            self.log.info(f"Uploading pairs -> {remote_base}/pairs/")
            self._rclone_sync(remote, f"{dest_path}/{config_id}/pairs", pairs_dir, flags)

        # 2) Upload comparison videos
        videos = list(self.config.data_root.glob("comparison_*.mp4"))
        if videos:
            vid_staging = self.config.data_root / "_videos"
            vid_staging.mkdir(exist_ok=True)
            for v in videos:
                shutil.copy2(v, vid_staging / v.name)
            self.log.info(f"Uploading {len(videos)} videos -> {remote_base}/videos/")
            self._rclone_sync(remote, f"{dest_path}/{config_id}/videos", vid_staging, flags)
            shutil.rmtree(vid_staging, ignore_errors=True)

        # 3) Upload checkpoints (nerfstudio logs)
        ns_dir = self.config.data_root / "logs" / "nerfstudio"
        if upload_ckpts and ns_dir.is_dir():
            self.log.info(f"Uploading checkpoints -> {remote_base}/checkpoints/")
            self._rclone_sync(remote, f"{dest_path}/{config_id}/checkpoints", ns_dir, flags)

        # 4) Delete local copies of what was uploaded; keep comparison videos on disk
        if delete_local:
            self.log.info("Deleting local data (keeping comparison_*.mp4 in scene root) ...")
            for name in ("raw", "shifted", "reverse_shifted", "rendered", "pairs"):
                d = self.config.data_root / name
                if d.is_dir():
                    self.log.info(f"  rm -rf {d}")
                    shutil.rmtree(d, ignore_errors=True)
            logs_root = self.config.data_root / "logs"
            if logs_root.is_dir():
                self.log.info(f"  rm -rf {logs_root}")
                shutil.rmtree(logs_root, ignore_errors=True)

        self.log.info("cleanup complete")

    # ── internal helpers ─────────────────────────────────────────────────────

    def _rclone_sync(self, remote: str, dest_path: str, src: Path,
                     flags: list[str] | None = None) -> None:
        cmd = ["rclone"]
        rclone_conf = self.param("rclone_config")
        if rclone_conf:
            conf_path = Path(rclone_conf).expanduser()
            if not conf_path.is_absolute():
                conf_path = self.config.repo_root / conf_path
            cmd.extend(["--config", str(conf_path)])
        cmd.extend(["copy", str(src), f"{remote}:{dest_path}"])
        if flags:
            cmd.extend(flags)
        self.sh(cmd)
