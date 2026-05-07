#!/usr/bin/env python3
"""
Run scenario pipeline from YAML config or scene CSV batches.

Usage:
    python run.py 01_spatad_cycle                               # base config only
    python run.py 01_spatad_cycle --config configs/scene_001.yaml  # with override
    python run.py 04_3dgs_dl3dv --scene-csv scenarios/04_3dgs_dl3dv/configs/scenes.csv --chunk-id 0
    python run.py 01_spatad_cycle --steps render save_pairs     # specific steps
    python run.py --list                                        # list scenarios
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.config import (
    PipelineConfig,
    SCENARIOS_DIR,
    load_scene_rows,
    make_config_id_from_scene_params,
)
from lib.loader import load_pipeline_class


def main():
    parser = argparse.ArgumentParser(description="Run a datagen scenario pipeline")
    parser.add_argument("scenario", nargs="?", help="Scenario name (e.g. 01_spatad_cycle)")
    parser.add_argument("--config", default=None, help="Path to per-job YAML config override")
    parser.add_argument("--scene-csv", default=None, help="Path to CSV with batch/scene/hash/chunk columns")
    parser.add_argument("--chunk-id", default=None, help="Filter scene CSV by chunk id value")
    parser.add_argument("--steps", nargs="*", default=None, help="Steps to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for d in sorted(SCENARIOS_DIR.iterdir()):
            if d.is_dir() and d.name != "_template" and d.name != "__pycache__":
                print(f"  {d.name}")
        return

    if not args.scenario:
        parser.print_help()
        sys.exit(1)

    PipelineCls = load_pipeline_class(args.scenario)
    if args.config and args.scene_csv:
        parser.error("Use only one input mode: --config OR --scene-csv.")

    if args.scene_csv:
        rows = load_scene_rows(args.scenario, args.scene_csv, int(args.chunk_id))

        for download_override in rows:
            config_id = make_config_id_from_scene_params(download_override)
            cfg = PipelineConfig.load_from_params(
                args.scenario,
                params_override={"download": download_override},
                config_id=config_id,
            )
            pipeline = PipelineCls(cfg)
            pipeline.run(args.steps)
        return

    cfg = PipelineConfig.load(args.scenario, args.config)
    pipeline = PipelineCls(cfg)
    pipeline.run(args.steps)


if __name__ == "__main__":
    main()
