import argparse
import bdb
import importlib
import pdb
import shutil
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import exp


def train_epoch():
    global global_step

    net_model.train()
    net_model.hparams.is_training = True

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, data in enumerate(pbar):
        data = exp.to_target_device(data, args.device)
        optimizer.zero_grad()
        loss = net_model.training_step(data, batch_idx)
        loss.backward()
        net_model.on_after_backward()
        optimizer.step()
        scheduler.step()
        net_model.log('learning_rate', scheduler.get_last_lr()[0])
        pbar.set_postfix_str(f"Loss = {loss.item():.2f}")

        net_model.write_log(writer, global_step)
        global_step += 1


def validate_epoch():
    global metric_val_best

    net_model.eval()
    net_model.hparams.is_training = False

    pbar = tqdm(val_loader, desc='Validation')
    for batch_idx, data in enumerate(pbar):
        data = exp.to_target_device(data, args.device)
        with torch.no_grad():
            net_model.validation_step(data, batch_idx)

    log = net_model.write_log(writer, global_step)
    metric_val = log['val_loss']

    model_state = {
        'state_dict': net_model.state_dict(),
        'epoch': epoch_idx, 'val_loss': metric_val
    }

    if metric_val < metric_val_best:
        metric_val_best = metric_val
        torch.save(model_state, train_log_dir / f"best.pth")
    torch.save(model_state, train_log_dir / f"newest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synorim Training script')
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to run on.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    args = parser.parse_args()

    exp.seed_everything(0)

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
    optimizers, schedulers = net_model.configure_optimizers()
    assert len(optimizers) == 1 and len(schedulers) == 1
    optimizer, scheduler = optimizers[0], schedulers[0]
    assert scheduler['interval'] == 'step'
    scheduler = scheduler['scheduler']

    # TensorboardX writer
    tb_logdir = train_log_dir / "tensorboard"
    tb_logdir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    # Move to target device
    args.device = torch.device(args.device)
    net_model = exp.to_target_device(net_model, args.device)
    net_model.device = args.device

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
