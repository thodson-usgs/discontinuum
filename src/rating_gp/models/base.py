from __future__ import annotations

from discontinuum.engines.base import DataMixin, ModelConfig
from discontinuum.pipeline import TimePipeline, UnitPipeline


class RatingDataMixin(DataMixin):
    """Data manager configuration for rating curve models."""

    def build_datamanager(self, model_config: ModelConfig | None = None):
        if model_config is None:
            model_config = ModelConfig()
        self._build_datamanager(
            covariate_pipelines={
                "time": TimePipeline,
                "stage": UnitPipeline,
            },
            model_config=model_config,
        )
