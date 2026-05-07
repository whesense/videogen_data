"""
Pipeline configuration: YAML loading, deep merging, and path resolution.

Config layering:
  1. scenarios/<name>/base.yaml          — scenario defaults
  2. scenarios/<name>/configs/<job>.yaml  — per-job overrides (deep-merged on top)

The merged dict is wrapped in PipelineConfig with computed paths.
"""
from __future__ import annotations

import copy
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCENARIOS_DIR = REPO_ROOT / "scenarios"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`. Lists are replaced, not appended."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


@dataclass
class PipelineConfig:
    scenario: str
    config_id: str
    repo_root: Path
    data_root: Path
    raw_dir: Path
    rendered_dir: Path
    pairs_dir: Path
    log_dir: Path
    params: dict = field(default_factory=dict)
    job_config_path: Path | None = None  # set when load() was given a job YAML path

    @classmethod
    def load(cls, scenario: str, config_file: str | Path | None = None) -> PipelineConfig:
        """Load and merge YAML configs, resolve paths."""
        load_base_params(scenario)
        config_id = "default"
        job_config_path: Path | None = None
        params_override: dict[str, Any] | None = None
        if config_file is not None:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = REPO_ROOT / config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            job_config_path = config_path.resolve()
            params_override = yaml.safe_load(config_path.read_text()) or {}
            config_id = config_path.stem
        return cls.load_from_params(
            scenario=scenario,
            params_override=params_override,
            config_id=config_id,
            job_config_path=job_config_path,
        )

    @classmethod
    def load_from_params(
        cls,
        scenario: str,
        params_override: dict[str, Any] | None = None,
        *,
        config_id: str = "default",
        job_config_path: Path | None = None,
    ) -> PipelineConfig:
        """Build config from base.yaml + in-memory override params."""
        params = load_base_params(scenario)
        if params_override:
            params = deep_merge(params, params_override)

        # Allow config to explicitly set config_id
        config_id = params.pop("config_id", config_id)

        # Resolve data root — overridable via "data_dir" in YAML
        custom_data_dir = params.pop("data_dir", None)
        if custom_data_dir:
            data_root = Path(custom_data_dir).expanduser()
            if not data_root.is_absolute():
                data_root = REPO_ROOT / data_root
            data_root = data_root / scenario / config_id
        else:
            data_root = DATA_DIR / scenario / config_id

        raw_dir = data_root / "raw"
        rendered_dir = data_root / "rendered"
        pairs_dir = data_root / "pairs"
        log_dir = data_root / "logs"

        for d in (raw_dir, rendered_dir, pairs_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)

        return cls(
            scenario=scenario,
            config_id=config_id,
            repo_root=REPO_ROOT,
            data_root=data_root,
            raw_dir=raw_dir,
            rendered_dir=rendered_dir,
            pairs_dir=pairs_dir,
            log_dir=log_dir,
            params=params,
            job_config_path=job_config_path,
        )

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested dict access: config.get("download", "remote") → params["download"]["remote"]."""
        obj = self.params
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                return default
            if obj is None:
                return default
        return obj


def make_config_id_from_scene_params(params: dict) -> str:
    """Deterministic filesystem-safe config id for CSV-sourced jobs."""
    parts = [f"{k}_{params[k]}" for k in sorted(params.keys())]
    return "_".join(parts)

def load_base_params(scenario: str) -> dict[str, Any]:
    """Load base.yaml for scenario, validating scenario existence."""
    scenario_dir = SCENARIOS_DIR / scenario
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario not found: {scenario_dir}")
    base_yaml = scenario_dir / "base.yaml"
    if not base_yaml.exists():
        return {}
    return yaml.safe_load(base_yaml.read_text()) or {}

def load_scene_rows(
    scenario: str,
    scene_csv: str | Path,
    chunk_id: int,
) -> list[dict[str, Any]]:
    """Load CSV rows and produce per-row download overrides for base.yaml null keys."""
    base = load_base_params(scenario)
    download = base.get("download") or {}
    if not isinstance(download, dict):
        raise ValueError(f"Invalid download section in base.yaml for scenario '{scenario}'")
    required_download_keys = [k for k, v in download.items() if v is None]

    df = pd.read_csv(scene_csv)
    df_chunk = df[df["chunk"] == chunk_id]
    rows_download_override = []
    for i, row in df_chunk.iterrows():
        download_override: dict[str, str] = {}
        for key in required_download_keys:
            val = row.get(key)
            if val is None:
                raise ValueError(
                    f"CSV row missing required value for download.{key}: {scene_csv}"
                )
            download_override[key] = val
        rows_download_override.append(download_override)
    return rows_download_override


def discover_configs(scenario: str, configs_dir: str | Path | None = None) -> list[Path]:
    """Find all .yaml config files for a scenario."""
    if configs_dir:
        d = Path(configs_dir)
    else:
        d = SCENARIOS_DIR / scenario / "configs"
    configs = sorted(d.glob("*.yaml"))
    configs = [c for c in configs if c.name != "example.yaml"]
    return configs


def configs_for_part(configs: list[Path], part_id: int, num_parts: int) -> list[Path]:
    """Return the disjoint slice for ``part_id`` (1-based) when splitting into ``num_parts`` contiguous runs.

    Sorted ``configs`` order is preserved; part 1 is the first chunk, part ``num_parts`` the last.
    Sizes differ by at most one when the count is not divisible by ``num_parts``.
    """
    if num_parts < 1:
        raise ValueError("num_parts must be >= 1")
    if not (1 <= part_id <= num_parts):
        raise ValueError(f"part_id must be in [1, {num_parts}], got {part_id}")
    n = len(configs)
    if n == 0:
        return []
    base, rem = divmod(n, num_parts)
    start = 0
    for k in range(part_id - 1):
        start += base + (1 if k < rem else 0)
    length = base + (1 if (part_id - 1) < rem else 0)
    return configs[start : start + length]
