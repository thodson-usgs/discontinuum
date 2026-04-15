from __future__ import annotations

from discontinuum.engines.base import DataMixin, ModelConfig
from discontinuum.pipeline import LogStandardPipeline, TimePipeline


class LoadestDataMixin(DataMixin):
    """Data manager configuration for load estimation models."""

    def build_datamanager(self, model_config: ModelConfig | None = None):
        if model_config is None:
            model_config = ModelConfig()
        self._build_datamanager(
            covariate_pipelines={
                "time": TimePipeline,
                "flow": LogStandardPipeline,
            },
            model_config=model_config,
        )
