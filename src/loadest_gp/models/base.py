from discontinuum.data_manager import DataManager
from discontinuum.pipeline import LogStandardPipeline, TimePipeline


class LoadestDataMixin:
    """ """
    # TODO inheret a BaseModel class with abc methods like build_model
    def build_datamanager(self):
        covariate_pipelines = {
            "time": TimePipeline,
            "flow": LogStandardPipeline
        }

        target_pipeline = LogStandardPipeline

        self.dm = DataManager(
            target_pipeline=target_pipeline,
            covariate_pipelines=covariate_pipelines
        )
