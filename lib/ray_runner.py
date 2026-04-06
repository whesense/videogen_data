"""
Ray-based distributed runner for datagen scenario pipelines.

Submits one Ray task per YAML config file.
Each task instantiates the scenario's Pipeline class and calls pipeline.run().

Usage:
    python ray_run.py 01_spatad_cycle
    python ray_run.py 01_spatad_cycle --ray-address auto   # cluster; set RAY_ADDRESS=host:6379 on driver
    python ray_run.py 01_spatad_cycle --max-parallel 0   # unlimited driver cap (default: 8)
    python ray_run.py 01_spatad_cycle --steps render save_pairs
    python ray_run.py 02_neurad_cycle --num-parts 4 --part-id 1   # disjoint chunk 1 of 4 (sorted configs)
    python ray_run.py 01_spatad_cycle --dry-run

Multi-node: start head/worker via ./ray_cluster.sh, export RAY_ADDRESS=<head_ip>:6379, then run with
--ray-address auto (or omit --ray-address if RAY_ADDRESS is already exported).
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_pipeline_task(repo_root: str, scenario: str, config_file: str,
                       steps: list[str] | None = None) -> dict:
    """Executed on a Ray worker: load config, import pipeline, run."""
    import os, sys
    root = Path(repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    os.chdir(root)

    from lib.config import PipelineConfig
    from lib.loader import load_pipeline_class

    config_id = Path(config_file).stem
    t0 = time.time()
    try:
        cfg = PipelineConfig.load(scenario, config_file)
        PipelineCls = load_pipeline_class(scenario)
        pipeline = PipelineCls(cfg)
        pipeline.run(steps)
        elapsed = time.time() - t0
        return {"config": cfg.config_id, "returncode": 0, "elapsed_s": round(elapsed, 1)}
    except Exception:
        elapsed = time.time() - t0
        return {
            "config": config_id,
            "returncode": 1,
            "elapsed_s": round(elapsed, 1),
            "error": traceback.format_exc(),
        }


def run_distributed(scenario: str, configs: list[Path],
                    ray_address: str | None = None,
                    max_parallel: int = 8,
                    steps: list[str] | None = None,
                    num_gpus: float = 1.0,
                    ray_run_part: str | None = None) -> list[dict]:
    """Submit all configs as Ray tasks and collect results.

    ``max_parallel`` defaults to 8 (cluster-wide in-flight cap). Use 0 to submit
    all tasks immediately and rely only on Ray resource limits.

    With num_gpus=1, Ray schedules at most one task per GPU cluster-wide (each
    worker gets CUDA_VISIBLE_DEVICES set to a single device).

    Forwards ``NEURAD_NUM_ITER`` / ``SPLATAD_NUM_ITER`` from the driver environment
    to workers so job scripts (e.g. ``run_jobs/run_neurad.sh``) can set short runs.

    Sets ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`` on workers (default ``python``) so
    NeuRAD / older generated protos work with protobuf 4–6 without regenerating stubs;
    slower than C++ parsing. Override on the driver: ``export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp``.
    """
    import os

    import ray

    env_vars = {k: os.environ[k] for k in ("NEURAD_NUM_ITER", "SPLATAD_NUM_ITER") if k in os.environ}
    env_vars["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = os.environ.get(
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python"
    )
    if ray_run_part:
        env_vars["RAY_RUN_PART"] = ray_run_part
    remote_kw: dict = {"num_gpus": num_gpus}
    remote_kw["runtime_env"] = {"env_vars": env_vars}
    run_task = ray.remote(**remote_kw)(_run_pipeline_task)

    if ray_address:
        ctx = ray.init(address=ray_address)
    else:
        ctx = ray.init()
    print(f"Ray cluster: {ray.cluster_resources()}")
    dashboard_url = None
    if ctx is not None:
        dashboard_url = getattr(ctx, "dashboard_url", None) or getattr(ctx, "webui_url", None)
    if not dashboard_url:
        try:
            dashboard_url = ray.get_runtime_context().get_dashboard_url()
        except Exception:
            pass
    if dashboard_url:
        print(f"Ray dashboard: {dashboard_url}")
    part_note = f" (part {ray_run_part})" if ray_run_part else ""
    print(f"Submitting {len(configs)} jobs{part_note}...\n")

    repo = str(REPO_ROOT)

    if max_parallel > 0:
        pending, results = [], []
        it = iter(configs)
        for cfg in it:
            if len(pending) >= max_parallel:
                break
            pending.append(run_task.remote(repo, scenario, str(cfg), steps))
        for cfg in it:
            ready, pending = ray.wait(pending, num_returns=1)
            results.extend(ray.get(ready))
            _progress(len(results), len(configs))
            pending.append(run_task.remote(repo, scenario, str(cfg), steps))
        results.extend(ray.get(pending))
    else:
        futures = [run_task.remote(repo, scenario, str(c), steps) for c in configs]
        results = []
        while futures:
            ready, futures = ray.wait(futures, num_returns=min(len(futures), 4))
            results.extend(ray.get(ready))
            _progress(len(results), len(configs))

    return results


def print_results(results: list[dict]) -> int:
    """Print summary table. Returns number of failures."""
    print("\n\n===== RESULTS =====\n")
    failed = 0
    for r in sorted(results, key=lambda x: x["config"]):
        ok = r["returncode"] == 0
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        line = f"  [{status}] {r['config']}  ({r['elapsed_s']}s)"
        if "error" in r:
            line += f"\n         {r['error'].splitlines()[-1]}"
        print(line)

    total = len(results)
    print(f"\nTotal: {total}  |  Passed: {total - failed}  |  Failed: {failed}")
    return failed


def _progress(done: int, total: int):
    pct = done * 100 // total
    print(f"\r  Progress: {done}/{total} ({pct}%)", end="", flush=True)
