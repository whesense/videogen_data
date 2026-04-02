"""
Pipeline configuration: YAML loading, deep merging, and path resolution.

Config layering:
  1. scenarios/<name>/base.yaml          — scenario defaults
  2. scenarios/<name>/configs/<job>.yaml  — per-job overrides (deep-merged on top)

The merged dict is wrapped in PipelineConfig with computed paths.
"""
from __future__ import annotations

import copy
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
        scenario_dir = SCENARIOS_DIR / scenario
        if not scenario_dir.is_dir():
            raise FileNotFoundError(f"Scenario not found: {scenario_dir}")

        # Layer 1: base.yaml
        base_yaml = scenario_dir / "base.yaml"
        params: dict[str, Any] = {}
        if base_yaml.exists():
            params = yaml.safe_load(base_yaml.read_text()) or {}

        # Layer 2: per-job override
        config_id = "default"
        job_config_path: Path | None = None
        if config_file is not None:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                config_path = REPO_ROOT / config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            job_config_path = config_path.resolve()
            override = yaml.safe_load(config_path.read_text()) or {}
            params = deep_merge(params, override)
            config_id = config_path.stem

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


def discover_configs(scenario: str, configs_dir: str | Path | None = None) -> list[Path]:
    """Find all .yaml config files for a scenario."""
    if configs_dir:
        d = Path(configs_dir)
    else:
        d = SCENARIOS_DIR / scenario / "configs"
    configs = sorted(d.glob("*.yaml"))
    configs = [c for c in configs if c.name != "example.yaml"]
    return configs
