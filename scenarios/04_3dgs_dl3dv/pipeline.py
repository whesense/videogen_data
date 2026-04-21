"""
Pipeline for scenario 04_3dgs_dl3dv.

GT data: DL3DV scenes (downloaded via aws-cli)
Rendered output: model view renders from test camera poses.

Invariant: download -> render -> save_pairs -> cleanup
  (Skip already-uploaded jobs offline: ``scripts/filter_uploaded_scene_configs.py``.)
  render      – trains original + cycle SplatAD, renders shifted & reverse-shifted
  save_pairs  – validates pairs/{gt,corrupted} + comparison video
                (gt = first-shift ``gt-rgb`` renders, not raw camera JPEGs)
  cleanup     – uploads to S3 then deletes local intermediates; ``run()`` always invokes
                cleanup in ``finally`` after primary steps (success or failure).  Local
                delete also runs if S3 upload raises (``delete_after_upload``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
import zipfile
import os

from lib.base_pipeline import BasePipeline
from .utils import get_points, spatial_sequence_split, TRAIN, TEST, NONE


class Pipeline(BasePipeline):
    @property
    def _batch(self) -> str:
        return self.param("download", "batch")

    @property
    def _scene(self) -> str:
        return self.param("download", "scene")

    @property
    def _3dgs_dir(self) -> str:
        return os.path.join(self.config.repo_root, self.param(
            "render", "3dgs_dir", default="gaussian-splatting"
        ))

    @property
    def _save_dir(self) -> str:
        return os.path.join(
            self.config.repo_root, self._3dgs_dir, self.param("download", "save_dir")
        )

    def _env(self) -> dict[str, str]:
        """Environment variables passed to .sh."""
        return {
            "PATH": "$HOME/.local/bin:$PATH",
            "PATH": "/home/jovyan/.mlspace/envs/gaussian_splatting/bin:$PATH", #TODO
            "AWS_SHARED_CREDENTIALS_FILE": self.param("aws_credentials_file"),
            "AWS_CONFIG_FILE": self.param("aws_config_file"),
        }

    def download(self) -> None:
        """Download a single DL3DV scene from S3."""
        src_path = os.path.join(
            self.param("download", "src_path"), self._batch, f"{self._scene}.zip"
        )
        os.makedirs(self._save_dir, exist_ok=True)
        self.sh(
            [
                "aws",
                "s3",
                "cp",
                "--endpoint-url",
                "https://s3.cloud.ru",
                src_path,
                self._save_dir,
            ],
            env_extra=self._env(),
        )
        with zipfile.ZipFile(
            os.path.join(self._save_dir, f"{self._scene}.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(self._save_dir, self._scene))

    def _prepare_scene(self):
        transforms_path = os.path.join(self._save_dir, self._scene, "transforms.json")
        image_frames, points = get_points(transforms_path)
        self.image_frames = image_frames

        min_dist = 2.0
        min_threshold = 0.33
        max_iters = 3
        for i in range(max_iters):
            labels, info = spatial_sequence_split(
                points,
                min_dist=min_dist,
                n_test_sequences=3,
                test_window_size=10,
                n_trials=100,
                alpha_none=0.5,
                # random_state=50,
            )
            percent_train = (labels == TRAIN).sum() / len(labels)
            self.log.info(f"iter={i}, min_dist={min_dist}, percent_train={percent_train:.3f}")
            if percent_train >= min_threshold:
                break
            min_dist /= 2  # relax constraint
        else:
            self.log.info("Warning: couldn't reach desired train percentage")
        self.labels = labels
        # plot_split(points, labels, os.path.join(SAVE_DIR, args.hash, 'split.jpg'))

        with open(
            os.path.join(self._save_dir, self._scene, "undistort/sparse/0/test.txt"),
            "w",
        ) as f:
            f.write(
                "\n".join(
                    [
                        image_frames[i]
                        for i, label in enumerate(labels)
                        if label == TEST or label == NONE
                    ]
                )
            )

        self.log.info("Successfully splitted camera poses")

    def render(self) -> None:
        """Run full 3dgs cycle."""
        self._prepare_scene()
        
        self.sh(
            [
                "python",
                "train.py",
                "-m",
                os.path.join("output", self._scene),
                "-s", 
                os.path.join(self._save_dir, self._scene, 'undistort'),
                "--eval",
                "--llffhold",
                "0",
                "--iterations",
                self.param("render", "iters")[-1],
                "--save_iterations",
                *self.param('render', 'iters'),
            ],
            cwd=self._3dgs_dir,
        )
        self.log.info("Successfully optimized scene")

        # for rendering keep only test scenes(not none)
        with open(
            os.path.join(self._save_dir, self._scene, "undistort/sparse/0/test.txt"),
            "w",
        ) as f:
            f.write(
                "\n".join(
                    [
                        self.image_frames[i]
                        for i, label in enumerate(self.labels)
                        if label == TEST
                    ]
                )
            )

        for iter in self.param('render', 'iters'):
            self.sh(
                ["python", "render.py", "-m", os.path.join("output", self._scene),
                "--iteration", iter, "--skip_train"],
                cwd=self._3dgs_dir,
            )


    def save_pairs(self) -> None:
        """Validate pairs/{gt,corrupted} and comparison videos.

        ``gt`` is taken from the shifted scene tree: prefer
        ``shifted/.../sensors/cameras/<cam>/<timestamp>_gt-rgb.jpg``, else ``shifted/.../gt-rgb/<cam>/<timestamp>.jpg``.
        ``corrupted`` matches ``reverse_shifted/.../sensors/cameras/`` RGB. Steps 6–7 in run_cycle_av2_splatad.sh.
        """
        root = os.path.join(self._3dgs_dir, "output", self._scene)
        # Create target dirs
        os.makedirs(f"{root}/logs", exist_ok=True)
        os.makedirs(f"{root}/pairs/gt", exist_ok=True)

        for iter in self.param("render", "iters"):
            iter_str = f"{int(iter):04d}"
            src_ply = f"{root}/point_cloud/iteration_{iter}/point_cloud.ply"
            dst_ply = f"{root}/logs/point_cloud_{iter_str}.ply"
            shutil.copy2(src_ply, dst_ply)

            src_corr = f"{root}/test/{iter}/pairs/corrupted"
            dst_corr = f"{root}/pairs/corrupted_{iter_str}"
            shutil.copytree(src_corr, dst_corr, dirs_exist_ok=True)

            # --- gt (only copy once) ---
            src_gt = f"{root}/test/{iter}/pairs/gt"
            dst_gt = f"{root}/pairs/gt"
            if not os.path.isdir(dst_gt):
                shutil.copytree(src_gt, dst_gt, dirs_exist_ok=True)

        dst_path = os.path.join(self.param("save_pairs", "dst_path"), self._batch, self._scene)

        for target in ['pairs', 'logs']:
            self.sh(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--endpoint-url",
                    "https://s3.cloud.ru",
                    f"{root}/{target}",
                    f"{dst_path}/{target}",
                    "--recursive",
                ],
                env_extra=self._env(),
            )    

    def cleanup(self) -> None:
        """Upload results to S3 and delete local intermediates.
        Upload failures do not block local deletion (when ``delete_after_upload`` is true).
        Job config YAML is removed only if the full upload phase completed without error.
        """
        zip_file = os.path.join(self._save_dir, f"{self._scene}.zip")
        if os.path.exists(zip_file):
            os.remove(zip_file)
        zip_dir = os.path.join(self._save_dir, self._scene)
        if os.path.isdir(zip_dir):
            shutil.rmtree(zip_dir)
        exp_dir = os.path.join(self._3dgs_dir, "output", self._scene)
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir)
        self.log.info(f'Successfully cleaned artifacts')
