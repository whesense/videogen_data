#!/usr/bin/env python3
"""
Run a single scenario pipeline (optionally with a specific config).

Usage:
    python run.py 01_spatad_cycle                               # base config only
    python run.py 01_spatad_cycle --config configs/scene_001.yaml  # with override
    python run.py 01_spatad_cycle --steps render save_pairs     # specific steps
    python run.py --list                                        # list scenarios
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.config import PipelineConfig, SCENARIOS_DIR
from lib.loader import load_pipeline_class


def main():
    parser = argparse.ArgumentParser(description="Run a datagen scenario pipeline")
    parser.add_argument("scenario", nargs="?", help="Scenario name (e.g. 01_spatad_cycle)")
    parser.add_argument("--config", default=None, help="Path to per-job YAML config override")
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

    cfg = PipelineConfig.load(args.scenario, args.config)
    PipelineCls = load_pipeline_class(args.scenario)
    pipeline = PipelineCls(cfg)
    pipeline.run(args.steps)


if __name__ == "__main__":
    main()
