import jittor
import bdb, traceback, pdb
import importlib
import argparse
from pathlib import Path
from utils import exp
from tqdm import tqdm


def test_epoch():
    net_model.eval()
    net_model.hparams.is_training = False

    pbar = tqdm(test_loader, desc='Test')
    meter = exp.AverageMeter()
    for batch_idx, data in enumerate(pbar):
        with jittor.no_grad():
            test_result, test_metric = net_model.test_step(data, batch_idx)

        meter.append_loss(test_metric)

    return meter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synorim Evaluation script')
    parser.add_argument('config', type=str, help='Path to the config file.')
    args = parser.parse_args()

    model_args = exp.parse_config_yaml(Path(args.config))
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)

    # Load dataset
    test_loader = net_model.test_dataloader()

    # Run test
    try:
        test_meter = test_epoch()
    except Exception as ex:
        if not isinstance(ex, bdb.BdbQuit):
            traceback.print_exc()
            pdb.post_mortem(ex.__traceback__)
        exit()

    print(test_meter.get_mean_loss_dict())
