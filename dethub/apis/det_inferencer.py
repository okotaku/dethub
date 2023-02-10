import os.path as osp
from typing import Tuple

from mmengine.config import Config
from mmengine.infer.infer import BaseInferencer
from mmengine.utils import get_installed_path, is_installed

from mmdet.apis import DetInferencer as Base


class DetInferencer(Base):

    def _load_model_from_metafile(self, model: str) -> Tuple[Config, str]:
        """Load config and weights from metafile.

        Args:
            model (str): model name defined in metafile.
        Returns:
            Tuple[Config, str]: Loaded Config and weights path defined in
            metafile.
        """
        model = model.lower()

        assert self.scope is not None, (
            'scope should be initialized if you want '
            'to load config from metafile.')
        project = 'dethub'
        assert is_installed(project), f'Please install {project}'
        package_path = get_installed_path(project)
        for model_cfg in BaseInferencer._get_models_from_package(package_path):
            model_name = model_cfg['Name'].lower()
            model_aliases = model_cfg.get('Alias', [])
            if isinstance(model_aliases, str):
                model_aliases = [model_aliases.lower()]
            else:
                model_aliases = [alias.lower() for alias in model_aliases]
            if (model_name == model or model in model_aliases):
                cfg = Config.fromfile(
                    osp.join(package_path, '.mim', model_cfg['Config']))
                weights = model_cfg['Weights']
                weights = weights[0] if isinstance(weights, list) else weights
                return cfg, weights
        raise ValueError(f'Cannot find model: {model} in {project}')
