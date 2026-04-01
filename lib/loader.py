"""Dynamic scenario loader — imports Pipeline class from scenarios/<name>/pipeline.py."""
from __future__ import annotations

import importlib
from typing import Type

from lib.base_pipeline import BasePipeline


def load_pipeline_class(scenario: str) -> Type[BasePipeline]:
    """Import and return the Pipeline class from a scenario module."""
    module_path = f"scenarios.{scenario}.pipeline"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(
            f"Cannot load scenario '{scenario}': "
            f"expected module at scenarios/{scenario}/pipeline.py"
        ) from e

    cls = getattr(mod, "Pipeline", None)
    if cls is None:
        raise AttributeError(
            f"scenarios/{scenario}/pipeline.py must define a 'Pipeline' class"
        )
    if not issubclass(cls, BasePipeline):
        raise TypeError(
            f"Pipeline class in '{scenario}' must inherit from BasePipeline"
        )
    return cls
