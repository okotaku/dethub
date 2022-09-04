import copy
import logging

from mmengine.hooks.ema_hook import EMAHook as Base
from mmengine.logging import print_log
from mmengine.registry import HOOKS
from mmengine.runner.checkpoint import _load_checkpoint_to_model


@HOOKS.register_module()
class EMAHook(Base):

    def after_load_checkpoint(self,
                              runner,
                              checkpoint: dict,
                              revise_keys: list = [(r'^module.', '')]) -> None:
        """Resume ema parameters from checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        if 'ema_state_dict' in checkpoint:
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            _load_checkpoint_to_model(
                self.ema_model.module,
                checkpoint['ema_state_dict'],
                False,
                revise_keys=revise_keys)

        # Support load checkpoint without ema state dict.
        else:
            print_log(
                'There is no `ema_state_dict` in checkpoint. '
                '`EMAHook` will make a copy of `state_dict` as the '
                'initial `ema_state_dict`', 'current', logging.WARNING)
            _load_checkpoint_to_model(
                self.ema_model.module,
                copy.deepcopy(checkpoint['state_dict']),
                False,
                revise_keys=revise_keys)
