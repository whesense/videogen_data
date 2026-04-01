"""
Abstract base class for all datagen scenario pipelines.

Each scenario implements a subclass and overrides download / render / save_pairs.
The base class provides logging, config access, and the step runner.

Typical scenario pipeline.py:

    from lib.base_pipeline import BasePipeline

    class Pipeline(BasePipeline):
        def download(self):
            ...
        def render(self):
            ...
        def save_pairs(self):
            ...
        def cleanup(self):
            ...
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lib.config import PipelineConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-5s %(asctime)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

STEPS = ("download", "render", "save_pairs", "cleanup")


class BasePipeline(ABC):
    """Override download / render / save_pairs / cleanup in your scenario."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.log = logging.getLogger(f"{config.scenario}/{config.config_id}")
        self._print_header()

    def _print_header(self):
        c = self.config
        self.log.info(f"Scenario  : {c.scenario}")
        self.log.info(f"Config    : {c.config_id}")
        self.log.info(f"Data root : {c.data_root}")
        self.log.info(f"Raw dir   : {c.raw_dir}")
        self.log.info(f"Rendered  : {c.rendered_dir}")
        self.log.info(f"Pairs     : {c.pairs_dir}")

    # ── Steps (override these) ──────────────────────────────────────────────

    @abstractmethod
    def download(self) -> None:
        """Step 1: Download ground-truth data."""
        ...

    @abstractmethod
    def render(self) -> None:
        """Step 2: Render corrupted / augmented frames."""
        ...

    @abstractmethod
    def save_pairs(self) -> None:
        """Step 3: Organize GT + rendered into paired dataset."""
        ...

    def cleanup(self) -> None:
        """Step 4: Upload results to remote storage and free local disk.

        Default implementation does nothing. Override in your scenario to
        upload pairs / checkpoints to S3 and remove intermediate files.
        """
        self.log.info("cleanup(): no-op (override in scenario to enable)")

    # ── Runner ──────────────────────────────────────────────────────────────

    def run(self, steps: list[str] | None = None) -> None:
        """Execute pipeline steps in order."""
        steps = steps or list(STEPS)
        for step_name in steps:
            if step_name not in STEPS:
                raise ValueError(f"Unknown step '{step_name}'. Valid: {STEPS}")
            self.log.info(f"{'=' * 10} {step_name} {'=' * 10}")
            method = getattr(self, step_name)
            method()
        self.log.info("Pipeline complete.")

    # ── Helpers available to all scenarios ───────────────────────────────────

    def sh(self, cmd: str | list[str], env_extra: dict[str, str] | None = None,
           **kwargs: Any) -> subprocess.CompletedProcess:
        """Run a shell command with logging. Raises on failure by default.

        Args:
            env_extra: additional env vars merged on top of os.environ.
        """
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
        self.log.info(f"$ {cmd_str}")
        kwargs.setdefault("check", True)
        kwargs.setdefault("shell", isinstance(cmd, str))
        if env_extra:
            env = {**os.environ, **env_extra}
            kwargs.setdefault("env", env)
        return subprocess.run(cmd, **kwargs)

    def rclone_copy(self, remote: str, src_path: str, dest: Path,
                    flags: list[str] | None = None) -> None:
        """Convenience wrapper around rclone copy.

        Automatically adds --config if top-level rclone_config is set in YAML.
        """
        cmd = ["rclone"]
        rclone_conf = self.param("rclone_config")
        if rclone_conf:
            conf_path = Path(rclone_conf).expanduser()
            if not conf_path.is_absolute():
                conf_path = self.config.repo_root / conf_path
            cmd.extend(["--config", str(conf_path)])
        cmd.extend(["copy", f"{remote}:{src_path}", str(dest)])
        if flags:
            cmd.extend(flags)
        self.sh(cmd)

    def param(self, *keys: str, default: Any = None) -> Any:
        """Shortcut for self.config.get(...)."""
        return self.config.get(*keys, default=default)
