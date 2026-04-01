#!/usr/bin/env python3
"""
Distributed pipeline runner — 1 Ray job per YAML config file.

Usage:
    python ray_run.py 01_spatad_cycle                         # all configs, local Ray
    python ray_run.py 01_spatad_cycle --ray-address auto      # existing cluster
    python ray_run.py 01_spatad_cycle --max-parallel 0        # no driver-side cap (default is 8)
    python ray_run.py 01_spatad_cycle --num-gpus 0            # CPU-only (no GPU reservation)
    python ray_run.py 01_spatad_cycle --steps render save_pairs
    python ray_run.py 01_spatad_cycle --dry-run               # preview jobs
    # While jobs run, open the printed "Ray dashboard:" URL (often http://127.0.0.1:8265).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.config import discover_configs, SCENARIOS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run scenario pipelines on a Ray cluster")
    parser.add_argument("scenario", help="Scenario name (e.g. 01_spatad_cycle)")
    parser.add_argument("--configs-dir", default=None, help="Custom configs directory")
    parser.add_argument("--ray-address", default=None, help="Ray cluster address")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Cluster-wide max concurrent jobs (default 8). Use 0 for no cap from the driver.",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=1.0,
        help="GPUs reserved per job (default 1.0). Use 0 for CPU-only. Ray limits concurrency to cluster GPU count.",
    )
    parser.add_argument("--steps", nargs="*", default=None, help="Steps to run (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="List configs without running")
    args = parser.parse_args()

    scenario_dir = SCENARIOS_DIR / args.scenario
    if not scenario_dir.is_dir():
        print(f"ERROR: Scenario not found: {scenario_dir}", file=sys.stderr)
        sys.exit(1)

    configs = discover_configs(args.scenario, args.configs_dir)
    if not configs:
        print(f"ERROR: No .yaml configs in {scenario_dir / 'configs'}", file=sys.stderr)
        sys.exit(1)

    print(f"Scenario : {args.scenario}")
    print(f"Configs  : {len(configs)} jobs")
    if args.steps:
        print(f"Steps    : {' → '.join(args.steps)}")
    print()

    if args.dry_run:
        for c in configs:
            print(f"  {c.stem}")
        print(f"\n{len(configs)} configs. Use without --dry-run to submit.")
        return

    from lib.ray_runner import run_distributed, print_results

    results = run_distributed(
        scenario=args.scenario,
        configs=configs,
        ray_address=args.ray_address,
        max_parallel=args.max_parallel,
        steps=args.steps,
        num_gpus=args.num_gpus,
    )
    failed = print_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
