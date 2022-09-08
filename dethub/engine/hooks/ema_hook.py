import copy
import logging

from mmengine.hooks.ema_hook import EMAHook as Base
from mmengine.logging import print_log
from mmengine.registry import HOOKS
from mmengine.runner.checkpoint import _load_checkpoint_to_model


@HOOKS.register_module(force=True)
class EMAHook(Base):

    def __init__(self,
                 ema_type: str = 'ExponentialMovingAverage',
                 strict_load: bool = True,
                 begin_iter: int = 0,
                 begin_epoch: int = 0,
                 **kwargs):
        self.strict_load = strict_load
        self.ema_cfg = dict(type=ema_type, **kwargs)
        assert not (begin_iter != 0 and begin_epoch != 0), (
            '`begin_iter` and `begin_epoch` should not be both set.')
        assert begin_iter >= 0, (
            f'begin_iter must larger than 0, but got begin: {begin_iter}')
        assert begin_epoch >= 0, (
            f'begin_epoch must larger than 0, but got begin: {begin_epoch}')
        self.begin_iter = begin_iter
        self.begin_epoch = begin_epoch
        # If `begin_epoch` and `begin_iter` are not set, `EMAHook` will be
        # enabled at 0 iteration.
        self.enabled_by_epoch = self.begin_epoch > 0

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
            runner (Runner): The runner of the testing process.
        """
        if 'ema_state_dict' in checkpoint and runner._resume:
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint['ema_state_dict'], strict=self.strict_load)

        # Support load checkpoint without ema state dict.
        else:
            print_log(
                'There is no `ema_state_dict` in checkpoint. '
                '`EMAHook` will make a copy of `state_dict` as the '
                'initial `ema_state_dict`', 'current', logging.WARNING)
            _load_checkpoint_to_model(self.ema_model.module,
                                      copy.deepcopy(checkpoint['state_dict']),
                                      False)
