from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

from discontinuum.data_manager import DataManager
from discontinuum.pipeline import (
    LogStandardPipeline,
    GeometricErrorPipeline,
    LogStandardErrorPipeline,
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
    error_type: Literal["gse", "se"] = "gse"


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
            if model_config.error_type == "gse":
                error_pipeline = GeometricErrorPipeline
            elif model_config.error_type == "se":
                error_pipeline = LogStandardErrorPipeline
            else:
                raise ValueError(
                    "Model config error_type must be 'gse' or 'se' when using "
                    "the 'log' transform."
                )
        elif model_config.transform == "standard":
            target_pipeline = StandardPipeline
            error_pipeline = StandardErrorPipeline
            if model_config.error_type != "se":
                raise ValueError(
                    "Model config error_type must be 'se' when using the "
                    "'standard' transform."
                )
        else:
            raise ValueError(
                "Model config transform must be 'log' or 'standard'."
            )

        self.dm = DataManager(
            target_pipeline=target_pipeline,
            error_pipeline=error_pipeline,
            covariate_pipelines=covariate_pipelines
        )
