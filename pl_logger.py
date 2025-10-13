import os
from clearml import Task
from lightning.pytorch.utilities import rank_zero_only

def _is_global_zero():
    return os.environ.get("NODE_RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"

class ClearMLLogger:
    """ClearML logger, compatible with pytorch_lightning.Trainer
    """
    def __init__(self, log_dir, project_name, task_name, tags):
        self.task_name = task_name
        self._task = None
        self._logger = None
        if _is_global_zero():  # <<< only rank-0 creates a Task
            self._task = Task.init(project_name=project_name, task_name=task_name, tags=tags, reuse_last_task_id=False)
            self._logger = self._task.get_logger()

        self.log_dir = log_dir

    @property
    def name(self) -> str:
        return self.task_name

    @property
    def version(self) -> str:
        return ""

    @property
    def save_dir(self):
        return self.log_dir

    @rank_zero_only
    def log_hyperparams(self, params):
        if self._task:
            self._task.connect(params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if not self._logger:
            return
        it = int(step or 0)
        for k, v in metrics.items():
            try:
                self._logger.report_scalar(title=k, series=k, value=float(v), iteration=it)
            except Exception:
                pass

    @rank_zero_only
    def finalize(self, status: str):
        pass # otherwise, the task will close after trainer.validate()
        # if self._task:
        #     self._task.close()

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback):
        pass

    @rank_zero_only
    def log_graph(self, model, input_array=None):
        pass
