"""
Pipeline template — copy this to a new scenario and fill in the steps.

    cp -r scenarios/_template scenarios/02_my_scenario
    # edit scenarios/02_my_scenario/pipeline.py
    # edit scenarios/02_my_scenario/base.yaml
    # add per-job configs in scenarios/02_my_scenario/configs/*.yaml
"""
from __future__ import annotations

from lib.base_pipeline import BasePipeline


class Pipeline(BasePipeline):

    def download(self) -> None:
        # remote = self.param("download", "remote")
        # src_path = self.param("download", "src_path")
        # self.rclone_copy(remote, src_path, self.config.raw_dir)
        self.log.warning("download() is a stub")

    def render(self) -> None:
        # self.sh(f"python scripts/render.py --input {self.config.raw_dir} --output {self.config.rendered_dir}")
        self.log.warning("render() is a stub")

    def save_pairs(self) -> None:
        # pair GT with rendered frames
        self.log.warning("save_pairs() is a stub")
