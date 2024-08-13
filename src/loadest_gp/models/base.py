from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

from discontinuum.data_manager import DataManager
from discontinuum.pipeline import (
    LogStandardPipeline,
    LogErrorPipeline,
    StandardPipeline,
    StandardErrorPipeline,
    TimePipeline,
)

if TYPE_CHECKING:
    from typing import Literal


@dataclass
class ModelConfig:
    """ """
    transform: Literal["log", "standard"] = "log"


class LoadestDataMixin:
    """ """
    # TODO inheret a BaseModel class with abc methods like build_model
    def build_datamanager(
            self,
            model_config: ModelConfig = ModelConfig(),
            ):
        """ """
        covariate_pipelines = {
            "time": TimePipeline,
            "flow": LogStandardPipeline
        }

        if model_config.transform == "log":
            target_pipeline = LogStandardPipeline
            error_pipeline = LogErrorPipeline
        elif model_config.transform == "standard":
            target_pipeline = StandardPipeline
            error_pipeline = StandardErrorPipeline
        else:
            raise ValueError(
                "Model config transform must be 'log' or 'standard'."
            )

        self.dm = DataManager(
            target_pipeline=target_pipeline,
            error_pipeline=error_pipeline,
            covariate_pipelines=covariate_pipelines
        )
