import pickle
from collections import OrderedDict

import sys
from pathlib import Path
from omegaconf import OmegaConf


def parse_config_yaml(yaml_path: Path, args: OmegaConf = None, override: bool = True) -> OmegaConf:
    """
    Load yaml file, and optionally merge it with existing ones.
    This supports a light-weight (recursive) inclusion scheme.
    :param yaml_path: path to the yaml file
    :param args: previous config
    :param override: if option clashes, whether or not to overwrite previous ones.
    :return: new config.
    """
    if args is None:
        args = OmegaConf.create()

    configs = OmegaConf.load(yaml_path)
    if "include_configs" in configs:
        base_configs = configs["include_configs"]
        del configs["include_configs"]
        if isinstance(base_configs, str):
            base_configs = [base_configs]
        # Update the config from top to down.
        for base_config in base_configs:
            base_config_path = yaml_path.parent / Path(base_config)
            configs = parse_config_yaml(base_config_path, configs, override=False)

    if "assign" in configs:
        overlays = configs["assign"]
        del configs["assign"]
        assign_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in overlays.items()])
        configs = OmegaConf.merge(configs, assign_config)

    if override:
        return OmegaConf.merge(args, configs)
    else:
        return OmegaConf.merge(configs, args)


class AverageMeter:
    """
    Maintain named lists of numbers. Compute their average to evaluate dataset statistics.
    This can not only used for loss, but also for progressive training logging, supporting import/export data.
    """
    def __init__(self):
        self.loss_dict = OrderedDict()

    def clear(self):
        self.loss_dict.clear()

    def export(self, f):
        if isinstance(f, str):
            f = open(f, 'wb')
        pickle.dump(self.loss_dict, f)

    def load(self, f):
        if isinstance(f, str):
            f = open(f, 'rb')
        self.loss_dict = pickle.load(f)
        return self

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val]})
            else:
                self.loss_dict[loss_name].append(loss_val)

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            loss_dict[loss_name] = sum(loss_arr) / len(loss_arr)
        return loss_dict

    def get_mean_loss(self):
        mean_loss_dict = self.get_mean_loss_dict()
        if len(mean_loss_dict) == 0:
            return 0.0
        else:
            return sum(mean_loss_dict.values()) / len(mean_loss_dict)

    def get_printable_mean(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, loss_mean in self.get_mean_loss_dict().items():
            all_loss_sum += loss_mean
            text += "(%s:%.4f) " % (loss_name, loss_mean)
        text += " sum = %.4f" % all_loss_sum
        return text

    def get_newest_loss_dict(self, return_count=False):
        loss_dict = {}
        loss_count_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            if len(loss_arr) > 0:
                loss_dict[loss_name] = loss_arr[-1]
                loss_count_dict[loss_name] = len(loss_arr)
        if return_count:
            return loss_dict, loss_count_dict
        else:
            return loss_dict

    def get_printable_newest(self):
        nloss_val, nloss_count = self.get_newest_loss_dict(return_count=True)
        return ", ".join([f"{loss_name}[{nloss_count[loss_name] - 1}]: {nloss_val[loss_name]}"
                          for loss_name in nloss_val.keys()])

    def print_format_loss(self, color=None):
        if hasattr(sys.stdout, "terminal"):
            color_device = sys.stdout.terminal
        else:
            color_device = sys.stdout
        if color == "y":
            color_device.write('\033[93m')
        elif color == "g":
            color_device.write('\033[92m')
        elif color == "b":
            color_device.write('\033[94m')
        print(self.get_printable_mean(), flush=True)
        if color is not None:
            color_device.write('\033[0m')

