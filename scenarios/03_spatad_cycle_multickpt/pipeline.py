"""
Pipeline for scenario 03_spatad_cycle_multickpt.

End-to-end order (``run.py`` / ``ray_run.py``): ``download`` → ``render`` → ``save_pairs`` → ``cleanup``.

Checkpoint behaviour (prune, remote latest-run sync) lives in :mod:`multickpt_utils`.

Steps: download → render → save_pairs → cleanup (upload + local delete).
"""
from __future__ import annotations

from pathlib import Path

from lib.base_pipeline import BasePipeline

from .multickpt_utils import (
    build_multickpt_pairs_layout,
    build_multickpt_render_env,
    clear_local_splatad_before_checkpoint_sync,
    copy_latest_splatad_checkpoint_run,
    delete_local_multickpt_artifacts,
    flatten_shifted_traj_remove_images_subdir,
    latest_splatad_config_yml,
    patch_checkpoint_configs_for_local_data,
    resolve_rclone_config_path,
    rclone_upload_copy_command,
)


class Pipeline(BasePipeline):

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
        return self.param("download", "remote", default=None) or self.param("remote", default=None)

    def _cleanup_remote(self) -> str | None:
        return self.param("cleanup", "remote", default=None) or self.param("remote", default=None)

    def _env(self) -> dict[str, str]:
        ns_dir = self.config.data_root / "logs" / "nerfstudio"
        extra = build_multickpt_render_env(
            ns_dir,
            self.config.config_id,
            self.config.data_root,
            self._scene,
            self.log,
        )
        # Absolute scene job root (…/data/03_spatad_cycle_multickpt/<config_id>/); 
        scene_root = str(self.config.data_root.resolve())
        return {
            "SEQ": self._scene,
            "CONFIG_ID": self.config.config_id,
            "SCENE_ROOT": scene_root,
            "REPO_DATA": str(self.config.data_root.parent.resolve()),
            "SHIFT": self.param("render", "shift", default="-3.0 0.0 0.0"),
            "GPU": str(self.param("render", "gpu", default="0")),
            "AV2_SENSOR_SPLIT": self.param("render", "av2_sensor_split", default="train"),
            **extra,
        }

    def download(self) -> None:
        remote = self._download_remote()
        if not remote:
            raise ValueError("Set download.remote (or legacy top-level remote) for rclone source")
        src_path = self.param("download", "src_path")
        scene = self._scene
        flags = self.param("download", "flags", default=[])
        split = self.param("download", "split", default=None)
        base = src_path.rstrip("/")
        full_path = f"{base}/{split}/{scene}"
        self.log.info(f"Scene  : {scene}")
        self.log.info(f"Source : {remote}:{full_path}")
        self.log.info(f"Dest   : {self.config.raw_dir}")
        self.rclone_copy(remote, full_path, self.config.raw_dir, flags)

        ck = self.param("download", "checkpoints_from", default=None)
        if ck:
            ck_remote = ck.get("remote") or self.param("remote", default=None)
            dest_path = ck.get("dest_path")
            if not ck_remote or not dest_path:
                raise ValueError("checkpoints_from requires remote and dest_path")
            ck_flags = ck.get("flags", []) or []
            cfg_id = self.config.config_id
            checkpoints_prefix = f"{dest_path.rstrip('/')}/{cfg_id}/checkpoints"
            dst_ck = self.config.data_root / "logs" / "nerfstudio"
            dst_ck.mkdir(parents=True, exist_ok=True)
            rclone_conf = resolve_rclone_config_path(
                self.config.repo_root, self.param("rclone_config")
            )
            for exp in (f"full_av2_{cfg_id}", f"cycle_av2_{cfg_id}"):
                clear_local_splatad_before_checkpoint_sync(dst_ck, exp, self.log)
                copy_latest_splatad_checkpoint_run(
                    ck_remote,
                    checkpoints_prefix,
                    exp,
                    dst_ck,
                    ck_flags,
                    rclone_conf,
                    self.log,
                    lambda r, s, d, f: self.rclone_copy(r, s, d, f),
                )
            patch_checkpoint_configs_for_local_data(
                dst_ck, cfg_id, self.config.data_root, self._scene, self.log
            )

    def render(self) -> None:
        script = self.param("render", "cycle_recon_script", default="run_cycle_av2_splatad_multickpt.sh")
        cycle_dir = self._cycle_dir
        self.log.info(f"Cycle dir : {cycle_dir}")
        self.log.info(f"Script    : {script}")
        self.log.info(f"Scene     : {self._scene}")
        self.sh(
            f"bash {script}",
            cwd=cycle_dir,
            env_extra=self._env(),
        )

    def save_pairs(self) -> None:
        ns_dir = self.config.data_root / "logs" / "nerfstudio"
        cycle_splatad = ns_dir / f"cycle_av2_{self.config.config_id}" / "splatad"
        step4_cfg = latest_splatad_config_yml(cycle_splatad)
        cycle_run_dir = step4_cfg.parent if step4_cfg is not None else None
        build_multickpt_pairs_layout(
            self.config.data_root,
            self._scene,
            self.param("render", "av2_sensor_split", default="train"),
            cycle_run_dir,
            self.log,
        )

        pairs = self.config.pairs_dir
        st = self.config.data_root / "shifted_traj"
        if st.is_dir():
            n_st = sum(1 for _ in st.rglob("*.jpg"))
            self.log.info(f"  shifted_traj (scene root): {n_st} images")
        else:
            self.log.warning("Missing shifted_traj/ under data root (render may not have run)")

        corrupted_dirs = sorted(pairs.glob("corrupted_*"))
        corrupted_dirs = [d for d in corrupted_dirs if d.is_dir()]
        for d in corrupted_dirs:
            n = sum(1 for _ in d.rglob("*.jpg"))
            self.log.info(f"  {d.name}: {n} images")

        gt = pairs / "gt"
        if gt.is_dir():
            n_gt = sum(1 for _ in gt.rglob("*.jpg"))
            self.log.info(f"  gt: {n_gt} images")

    def cleanup(self) -> None:
        remote = self._cleanup_remote()
        dest_path = self.param("cleanup", "dest_path")
        flags = self.param("cleanup", "flags", default=[])
        delete_local = self.param("cleanup", "delete_after_upload", default=True)
        delete_job_cfg = self.param("cleanup", "delete_job_config", default=False)

        upload_ok = False
        try:
            if not remote or not dest_path:
                self.log.warning(
                    "cleanup: cleanup.remote / dest_path not configured; skipping upload"
                )
            else:
                config_id = self.config.config_id
                remote_base = f"{remote}:{dest_path}/{config_id}"
                rc = self.param("rclone_config")

                pairs_dir = self.config.pairs_dir
                if pairs_dir.is_dir() and any(pairs_dir.iterdir()):
                    self.log.info(f"Uploading pairs (images) -> {remote_base}/pairs/")
                    self.sh(
                        rclone_upload_copy_command(
                            pairs_dir,
                            remote,
                            f"{dest_path}/{config_id}/pairs",
                            flags,
                            self.config.repo_root,
                            rc,
                        )
                    )

                st_extra = self.config.data_root / "shifted_traj"
                if st_extra.is_dir() and any(st_extra.iterdir()):
                    flatten_shifted_traj_remove_images_subdir(st_extra)
                    self.log.info(f"Uploading shifted_traj (images) -> {remote_base}/shifted_traj/")
                    self.sh(
                        rclone_upload_copy_command(
                            st_extra,
                            remote,
                            f"{dest_path}/{config_id}/shifted_traj",
                            flags,
                            self.config.repo_root,
                            rc,
                        )
                    )

                upload_ok = True
        except Exception:
            self.log.exception("cleanup: S3 upload phase failed")
        finally:
            if delete_local:
                delete_local_multickpt_artifacts(self.config.data_root, self.log)
            if upload_ok:
                self._delete_job_config_yaml(delete_job_cfg)
        self.log.info("cleanup complete")

    def _delete_job_config_yaml(self, enabled: bool) -> None:
        if not enabled or not self.config.job_config_path:
            return
        p = self.config.job_config_path
        cfg_root = self.config.repo_root / "scenarios" / self.config.scenario / "configs"
        try:
            p.resolve().relative_to(cfg_root.resolve())
        except ValueError:
            self.log.warning(f"cleanup: not deleting config outside scenario configs/: {p}")
            return
        if p.is_file():
            self.log.info(f"Removing job config {p}")
            p.unlink(missing_ok=True)
