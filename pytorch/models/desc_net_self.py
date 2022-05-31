from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.base import DatasetSpec as DS, list_collate
from dataset.flow_dataset import FlowDataset
from models.desc_net import Model as BaseModel


class Model(BaseModel):
    """
    Self-supervised setting of training the descriptor network.
    """
    @staticmethod
    def compute_self_sup_loss(pc0, pc1, pd_flow01, loss_config):
        pc0_warpped = pc0 + pd_flow01
        dist01 = torch.cdist(pc0_warpped, pc1)
        loss_dict = {}

        if loss_config.chamfer_weight > 0.0:
            chamfer01 = torch.min(dist01, dim=-1).values
            chamfer10 = torch.min(dist01, dim=-2).values
            loss_dict['chamfer'] = loss_config.chamfer_weight * (chamfer01.mean() + chamfer10.mean())

        if loss_config.laplacian_weight > 0.0:
            dist11 = torch.cdist(pc1, pc1)
            _, kidx1 = torch.topk(dist11, 10, dim=-1, largest=False, sorted=False)       # (N, 10)
            pc1_laplacian = torch.sum(pc1[kidx1] - pc1.unsqueeze(1), dim=1) / 9.0
            dist00 = torch.cdist(pc0, pc0)
            _, kidx0 = torch.topk(dist00, 10, dim=-1, largest=False, sorted=False)
            pc0_laplacian = torch.sum(pc0_warpped[kidx0] - pc0_warpped.unsqueeze(1), dim=1) / 9.0
            # Interpolate pc1_laplacian to pc0's
            dist, knn01 = torch.topk(dist01, 5, dim=-1, largest=False, sorted=False)
            pc1_lap_more = pc1_laplacian[knn01]
            norm = torch.sum(1.0 / (dist + 1.0e-6), dim=1, keepdim=True)
            weight = (1.0 / (dist + 1.0e-6)) / norm
            pc1_lap_0 = torch.sum(weight.unsqueeze(-1) * pc1_lap_more, dim=1)
            loss_dict['laplacian'] = loss_config.laplacian_weight * \
                                     torch.sum((pc1_lap_0 - pc0_laplacian) ** 2, dim=-1).mean()

        if loss_config.smooth_weight > 0.0:
            grouped_flow = pd_flow01[kidx0]     # (N, K, 3)
            loss_dict['smooth'] = loss_config.smooth_weight * \
                        (((grouped_flow - pd_flow01.unsqueeze(1)) ** 2).sum(-1).sum(-1) / 9.0).mean()

        return loss_dict

    def compute_loss(self, batch, desc_output, all_sels, compute_metric=False):
        num_batches = len(batch[DS.PC][0])
        loss_dict = defaultdict(list)
        loss_config = self.hparams.self_supervised_loss

        for batch_idx in range(num_batches):
            cur_pc0, cur_pc1 = batch[DS.PC][0][batch_idx], batch[DS.PC][1][batch_idx]
            cur_sel0, cur_sel1 = all_sels[batch_idx * 2 + 0], all_sels[batch_idx * 2 + 1]
            cur_feat0 = desc_output.features_at(batch_idx * 2 + 0)
            cur_feat1 = desc_output.features_at(batch_idx * 2 + 1)
            dist_mat = torch.cdist(cur_feat0, cur_feat1) / torch.maximum(
                torch.tensor(np.float32(self.hparams.td_min), device=self.device), self.td)
            cur_sub_pc0, cur_sub_pc1 = cur_pc0[cur_sel0], cur_pc1[cur_sel1]
            cur_pd0 = torch.softmax(-dist_mat, dim=1) @ cur_sub_pc1 - cur_sub_pc0
            cur_pd1 = torch.softmax(-dist_mat, dim=0).transpose(-1, -2) @ cur_sub_pc0 - cur_sub_pc1

            cur_loss_unsup01 = self.compute_self_sup_loss(cur_sub_pc0, cur_sub_pc1, cur_pd0, loss_config)
            cur_loss_unsup10 = self.compute_self_sup_loss(cur_sub_pc1, cur_sub_pc0, cur_pd1, loss_config)
            for ss_lkey in cur_loss_unsup01.keys():
                loss_dict[f'self-{ss_lkey}'].append(cur_loss_unsup01[ss_lkey])
                loss_dict[f'self-{ss_lkey}'].append(cur_loss_unsup10[ss_lkey])

        loss_dict = {k: sum(v) / len(v) for k, v in loss_dict.items()}
        return loss_dict, {}

    def get_dataset_spec(self):
        return [DS.FILENAME, DS.QUANTIZED_COORDS, DS.PC]

    def train_dataloader(self):
        train_set = FlowDataset(**self.hparams.train_kwargs, spec=self.get_dataset_spec(),
                                hparams=self.hparams, augmentor=None)
        torch.manual_seed(0)
        return DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=4, collate_fn=list_collate)
