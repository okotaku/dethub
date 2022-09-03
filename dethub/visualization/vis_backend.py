import os
from typing import Optional

from mmengine.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend as Base


@VISBACKENDS.register_module(force=True)
class WandbVisBackend(Base):

    def __init__(self,
                 save_dir: str,
                 init_kwargs: Optional[dict] = None,
                 define_metric_cfg: Optional[dict] = None,
                 commit: Optional[bool] = True):
        super(Base, self).__init__(save_dir)
        self._init_kwargs = init_kwargs
        self._define_metric_cfg = define_metric_cfg
        self._commit = commit

    def _init_env(self):
        """Setup env for wandb."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        wandb.init(**self._init_kwargs)
        if self._define_metric_cfg is not None:
            for metric, summary in self._define_metric_cfg.items():
                wandb.define_metric(f'val/{metric}', summary=summary)
        self._wandb = wandb
