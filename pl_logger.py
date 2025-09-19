from lightning.pytorch.utilities import rank_zero_only
from typing import List, Dict, Optional
from clearml import Task
import os
import json


class ClearMLLogger:
    """ClearML logger, compatible with pytorch_lightning.Trainer
    """
    def __init__(self, project_name, task_name, tags):
        super().__init__()
        self.task_name = task_name
        self._task = Task.init(project_name=project_name, task_name=task_name, tags=tags,
                               reuse_last_task_id=False)   
        self._logger = self._task.get_logger()
        # self._all_metrics = []

    @property
    def name(self) -> str:
        # return "PL_ClearMLLogger"
        return self.task_name

    @property
    def version(self) -> str:
        return ""

    @property
    def save_dir(self):
        return "./lightning_logs"

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback):
        pass

    @rank_zero_only
    def log_graph(self, model, input_array=None):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        self._task.connect(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for (name, value) in metrics.items():
            self._logger.report_scalar(title=name, series=name, value=value, iteration=step)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        # otherwise, the task is closed after trainer.validate call
        pass
