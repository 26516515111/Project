from __future__ import annotations

from typing import Any, Optional

from .pipeline import run_pipeline


class LanggraphyPipelineAdapter:
    """Adapter for running the LangGraph pipeline via a unified interface."""

    def __init__(self, default_args: Optional[dict[str, Any]] = None) -> None:
        self.default_args = default_args or {}

    def run(self, **kwargs: Any):
        """Run pipeline with merged default and call arguments."""

        merged = {**self.default_args, **kwargs}
        return run_pipeline(**merged)
