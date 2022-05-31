import functools
import importlib
import tempfile
from pathlib import Path
from typing import Mapping, Any, Optional, Callable, Union

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from utils.exp import AverageMeter, parse_config_yaml


def lambda_lr_wrapper(it, lr_config, batch_size):
    return max(
        lr_config['decay_mult'] ** (int(it * batch_size / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])


class BaseModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.log_cache = AverageMeter()

    @staticmethod
    def load_module(spec_path):
        """
        Load a module given spec_path
        :param spec_path: Path to a model ckpt.
        :return: the module class, possibly with weight loaded.
        """
        spec_path = Path(spec_path)
        config_args = parse_config_yaml(spec_path.parent / "config.yaml")
        net_module = importlib.import_module("models." + config_args.model).Model
        net_model = net_module(config_args)
        if "none.pth" not in spec_path.name:
            ckpt_data = torch.load(spec_path)
            net_model.load_state_dict(ckpt_data['state_dict'])
            print(f"Checkpoint loaded from {spec_path}.")
        return net_model

    def configure_optimizers(self):
        lr_config = self.hparams.learning_rate
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr_config['init'],
                                          weight_decay=self.hparams.weight_decay, amsgrad=True)
        else:
            raise NotImplementedError
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_after_backward(self):
        grad_clip_val = self.hparams.get('grad_clip', 1000.)
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=grad_clip_val)

        # Also remove nan values if any.
        has_nan_value = False
        for p in filter(lambda p: p.grad is not None, self.parameters()):
            pdata = p.grad.data
            grad_is_nan = pdata != pdata
            if torch.any(grad_is_nan):
                has_nan_value = True
                pdata[grad_is_nan] = 0.
        if has_nan_value:
            print(f"Warning: Gets a nan-gradient but set to 0.")

    def log(self, key, value):
        if self.hparams.is_training:
            assert key not in self.log_cache.loss_dict
        self.log_cache.append_loss({
            key: value.item() if isinstance(value, torch.Tensor) else value
        })

    def log_dict(self, dictionary: Mapping[str, Any]):
        for k, v in dictionary.items():
            self.log(str(k), v)

    def write_log(self, writer, it):
        logs_written = {}
        if not self.hparams.is_training or it % 10 == 0:
            for k, v in self.log_cache.get_mean_loss_dict().items():
                writer.add_scalar(k, v, it)
                logs_written[k] = v
        self.log_cache.clear()
        return logs_written
