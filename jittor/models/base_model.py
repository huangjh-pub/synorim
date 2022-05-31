import functools
import importlib
from pathlib import Path

import jittor
from jittor import nn
from jittor.optim import LambdaLR

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
        ckpt_data = jittor.load(str(spec_path))
        net_model.load_state_dict(ckpt_data['state_dict'])
        print(f"Checkpoint loaded from {spec_path}.")
        return net_model

    def configure_optimizers(self):
        lr_config = self.hparams.learning_rate
        optimizer = jittor.optim.AdamW(self.parameters(), lr=lr_config['init'],
                                       weight_decay=self.hparams.weight_decay)
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size))
        return optimizer, scheduler
