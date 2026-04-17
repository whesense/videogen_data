#!/usr/bin/env python3
"""
Distributed pipeline runner — 1 Ray job per YAML config file.

Usage:
    python ray_run.py 01_spatad_cycle                         # all configs, local Ray
    python scripts/generate_av2_scene_configs.py              # once: fill configs/ from rclone
    python ray_run.py 01_spatad_cycle --ray-address auto      # existing cluster (uses RAY_ADDRESS)
    python ray_run.py 01_spatad_cycle --max-parallel 0        # no driver-side cap (default is 8)
    python ray_run.py 01_spatad_cycle --num-gpus 0            # CPU-only (no GPU reservation)
    python ray_run.py 01_spatad_cycle --steps render save_pairs
    python ray_run.py 01_spatad_cycle --dry-run               # preview jobs
    python ray_run.py 03_spatad_cycle_multickpt --config scenarios/03_spatad_cycle_multickpt/configs/scene_022af476.yaml
    python ray_run.py 02_neurad_cycle --num-parts 4 --part-id 1   # 1/4 of configs (sorted, disjoint chunks)

Multi-node (2 machines): start Ray on each host, then run the driver on one host with RAY_ADDRESS set.

    # Node A (head):   ./ray_cluster.sh head
    # Node B (worker): RAY_HEAD_ADDRESS=<head_ip>:6379 ./ray_cluster.sh worker
    # Driver (often on head): export RAY_ADDRESS=<head_ip>:6379
    #                         ./ray_run.sh 01_spatad_cycle --ray-address auto --max-parallel 0

    # While jobs run, open the printed "Ray dashboard:" URL (often http://<head_ip>:8265).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.config import configs_for_part, discover_configs, SCENARIOS_DIR


def resolve_ray_address(cli: str | None) -> str | None:
    """Address for ``ray.init(address=...)``. None means a fresh local single-process cluster.

    Precedence: ``--ray-address`` > ``$RAY_ADDRESS`` (as ``auto``, so Ray reads the env) > local.
    """
    if cli is not None:
        return cli
    if os.environ.get("RAY_ADDRESS"):
        return "auto"
    return None


def main():
    parser = argparse.ArgumentParser(description="Run scenario pipelines on a Ray cluster")
    parser.add_argument("scenario", help="Scenario name (e.g. 01_spatad_cycle)")
    parser.add_argument("--configs-dir", default=None, help="Custom configs directory")
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray cluster: explicit host:port, or 'auto' (uses RAY_ADDRESS). "
        "If omitted and RAY_ADDRESS is set, 'auto' is implied.",
    )
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
    parser.add_argument(
        "--num-parts",
        type=int,
        default=None,
        metavar="N",
        help="Split sorted configs into N disjoint runs (use with --part-id).",
    )
    parser.add_argument(
        "--part-id",
        type=int,
        default=None,
        metavar="K",
        help="Run only part K of N (1-based). Requires --num-parts.",
    )
    args = parser.parse_args()

    if (args.part_id is None) ^ (args.num_parts is None):
        print(
            "ERROR: use --num-parts N and --part-id K together (e.g. --num-parts 4 --part-id 1).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.num_parts is not None:
        if args.num_parts < 1:
            print("ERROR: --num-parts must be >= 1", file=sys.stderr)
            sys.exit(1)
        if not (1 <= args.part_id <= args.num_parts):
            print(
                f"ERROR: --part-id must be between 1 and {args.num_parts}",
                file=sys.stderr,
            )
            sys.exit(1)

    scenario_dir = SCENARIOS_DIR / args.scenario
    if not scenario_dir.is_dir():
        print(f"ERROR: Scenario not found: {scenario_dir}", file=sys.stderr)
        sys.exit(1)

    configs = discover_configs(args.scenario, args.configs_dir)
    if not configs:
        print(f"ERROR: No .yaml configs in {scenario_dir / 'configs'}", file=sys.stderr)
        sys.exit(1)

    total_before_part = len(configs)
    ray_run_part: str | None = None
    if args.num_parts is not None and args.num_parts > 1:
        configs = configs_for_part(configs, args.part_id, args.num_parts)
        ray_run_part = f"{args.part_id}/{args.num_parts}"

    print(f"Scenario : {args.scenario}")
    if ray_run_part:
        print(f"Part     : {ray_run_part}  ({len(configs)} of {total_before_part} configs)")
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
        ray_address=resolve_ray_address(args.ray_address),
        max_parallel=args.max_parallel,
        steps=args.steps,
        num_gpus=args.num_gpus,
        ray_run_part=ray_run_part,
    )
    failed = print_results(results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
