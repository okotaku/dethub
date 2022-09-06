import os
from typing import Optional

import numpy as np
from mmengine.registry import VISBACKENDS
from mmengine.visualization import WandbVisBackend as Base
from mmengine.visualization.vis_backend import force_init_env


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
                wandb.define_metric(metric, summary=summary)
        self._wandb = wandb

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to wandb.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """
        image = self._wandb.Image(image)
        self._wandb.log({name: image}, commit=self._commit)
