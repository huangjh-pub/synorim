import argparse
import bdb
import importlib
import pdb
import shutil
import traceback
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

import jittor
from utils import exp

jittor.flags.use_cuda = True


def train_epoch():
    global global_step

    net_model.train()
    net_model.hparams.is_training = True

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, data in enumerate(pbar):
        optimizer.zero_grad()
        loss = net_model.training_step(data, batch_idx)
        optimizer.step(loss)
        scheduler.step()
        pbar.set_postfix_str(f"Loss = {loss.item():.2f}")

        global_step += 1


def validate_epoch():
    global metric_val_best

    net_model.eval()
    net_model.hparams.is_training = False

    pbar = tqdm(val_loader, desc='Validation')
    meter = exp.AverageMeter()
    for batch_idx, data in enumerate(pbar):
        with jittor.no_grad():
            val_loss = net_model.validation_step(data, batch_idx)
        meter.append_loss({'loss': val_loss.item()})

    metric_val = meter.get_mean_loss()
    model_state = {
        'state_dict': net_model.state_dict(),
        'epoch': epoch_idx, 'val_loss': metric_val
    }

    if metric_val < metric_val_best:
        metric_val_best = metric_val
        jittor.save(model_state, train_log_dir / f"best.jt")
    jittor.save(model_state, train_log_dir / f"newest.jt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synorim Training script')
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    args = parser.parse_args()

    model_args = exp.parse_config_yaml(Path(args.config))
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)

    train_log_dir = Path("out") / model_args.name
    train_log_dir.mkdir(exist_ok=True, parents=True)

    print(" >>>> ======= MODEL HYPER-PARAMETERS ======= <<<< ")
    print(OmegaConf.to_yaml(net_model.hparams, resolve=True))
    print("Save Directory is in:", train_log_dir)
    print(" >>>> ====================================== <<<< ")

    # Copy the model definition and config.
    shutil.copy(f"models/{model_args.model.replace('.', '/')}.py", train_log_dir / "model.py")
    OmegaConf.save(model_args, train_log_dir / "config.yaml")

    # Load dataset
    train_loader = net_model.train_dataloader()
    val_loader = net_model.val_dataloader()

    # Load training specs
    optimizer, scheduler = net_model.configure_optimizers()

    # Train and validate within a protected loop.
    global_step = 0
    metric_val_best = 1e6
    try:
        for epoch_idx in range(args.epochs):
            train_epoch()
            validate_epoch()
    except Exception as ex:
        if not isinstance(ex, bdb.BdbQuit):
            traceback.print_exc()
            pdb.post_mortem(ex.__traceback__)
