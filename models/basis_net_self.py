from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset.base import DatasetSpec as DS, list_collate
from dataset.flow_dataset import FlowDataset
from models.basis_net import Model as BaseModel
from models.desc_net_self import Model as DescModel


class Model(BaseModel):
    """
    Self-supervised setting of training the basis network.
    """
    def compute_loss(self, batch, basis_output, all_sels, compute_metric=False):
        num_batches = len(batch[DS.PC][0])
        loss_dict = defaultdict(list)
        loss_config = self.hparams.self_supervised_loss

        robust_kernel, robust_iter = self.get_robust_kernel()
        for batch_idx in range(num_batches):
            cur_pc0, cur_pc1 = batch[DS.PC][0][batch_idx], batch[DS.PC][1][batch_idx]
            cur_sel0, cur_sel1 = all_sels[batch_idx * 2 + 0], all_sels[batch_idx * 2 + 1]
            cur_sub_pc0, cur_sub_pc1 = cur_pc0[cur_sel0], cur_pc1[cur_sel1]
            cur_basis0 = basis_output[0].features_at(batch_idx * 2 + 0)
            cur_basis1 = basis_output[0].features_at(batch_idx * 2 + 1)
            cur_desc0 = basis_output[1].features_at(batch_idx * 2 + 0)
            cur_desc1 = basis_output[1].features_at(batch_idx * 2 + 1)

            aligned_basis0, aligned_basis1 = self.align_basis_via_pd_test(
                cur_basis0, cur_basis1, cur_desc0, cur_desc1, thres=0.6)
            c01 = self.solve_optimal_maps(aligned_basis0, aligned_basis1,
                                          robust_kernel=robust_kernel, num_iter=robust_iter)
            c10 = self.solve_optimal_maps(aligned_basis1, aligned_basis0,
                                          robust_kernel=robust_kernel, num_iter=robust_iter)

            if self.hparams.ctc_weight:
                ctc_err = c01 @ c10 - torch.eye(c01.size(0), device=self.device)
                loss_dict['ctc'].append(self.hparams.ctc_weight * torch.sum(ctc_err ** 2))

            coords0 = batch[DS.QUANTIZED_COORDS][0][0][batch_idx]
            coords1 = batch[DS.QUANTIZED_COORDS][1][0][batch_idx]
            flow01 = self.compute_flow_from_maps(cur_basis0, cur_basis1, c01,
                                                 cur_sub_pc0, cur_sub_pc1, sparse_coords=coords0)
            flow10 = self.compute_flow_from_maps(cur_basis1, cur_basis0, c10,
                                                 cur_sub_pc1, cur_sub_pc0, sparse_coords=coords1)

            for flow_type, flow_weight in zip(['final', 'f'], [1.0, self.hparams.flow_f_weight]):
                if flow_weight <= 0.0:
                    continue
                cur_loss_unsup01 = DescModel.compute_self_sup_loss(
                    cur_sub_pc0, cur_sub_pc1, flow01[flow_type], loss_config)
                cur_loss_unsup10 = DescModel.compute_self_sup_loss(
                    cur_sub_pc1, cur_sub_pc0, flow10[flow_type], loss_config)
                for ss_lkey in cur_loss_unsup01.keys():
                    loss_dict[f'self-{flow_type}-{ss_lkey}'].append(flow_weight * cur_loss_unsup01[ss_lkey])
                    loss_dict[f'self-{flow_type}-{ss_lkey}'].append(flow_weight * cur_loss_unsup10[ss_lkey])

        loss_dict = {k: sum(v) / len(v) for k, v in loss_dict.items()}
        return loss_dict, {}

    def get_dataset_spec(self):
        return [DS.QUANTIZED_COORDS, DS.PC]

    def train_dataloader(self):
        train_set = FlowDataset(**self.hparams.train_kwargs, spec=self.get_dataset_spec(),
                                hparams=self.hparams, augmentor=None)
        torch.manual_seed(0)
        return DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=4, collate_fn=list_collate)
