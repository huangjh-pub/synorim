import torch
import MinkowskiEngine as ME
from torch.nn import Parameter
from torch.utils.data import DataLoader

from dataset.base import DatasetSpec as DS, list_collate
from dataset.flow_dataset import FlowDataset, DataAugmentor
from metric import PairwiseFlowMetric

from models.spconv import ResUNet
from models.base_model import BaseModel
import numpy as np
from utils.point import propagate_features


class Model(BaseModel):
    """
    This model trains the descriptor network. This is the 1st stage of training.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.backbone_args = self.hparams.backbone_args
        self.backbone = ResUNet(self.backbone_args,
                                in_channels=3,
                                out_channels=self.backbone_args.out_channels,
                                normalize_feature=True,
                                conv1_kernel_size=3)
        self.td = Parameter(torch.tensor(np.float32(self.hparams.td_init)), requires_grad=True)

    def forward(self, batch):
        """
        Forward descriptor network.
            As the backbone quantized point cloud into voxels (by selecting one point for each voxel),
        we also return the selected point indices.
        """
        num_batches = len(batch[DS.QUANTIZED_COORDS][0][0])
        num_views = len(batch[DS.QUANTIZED_COORDS])
        all_coords, all_feats, all_sels = [], [], []
        for batch_idx in range(num_batches):
            for view_idx in range(num_views):
                all_coords.append(batch[DS.QUANTIZED_COORDS][view_idx][0][batch_idx])
                cur_sel = batch[DS.QUANTIZED_COORDS][view_idx][1][batch_idx]
                all_sels.append(cur_sel)
                all_feats.append(batch[DS.PC][view_idx][batch_idx][cur_sel])
        coords_batch, feats_batch = ME.utils.sparse_collate(all_coords, all_feats, device=self.device)
        sinput = ME.SparseTensor(feats_batch, coordinates=coords_batch)
        soutput = self.backbone(sinput)
        return soutput, all_sels

    def compute_loss(self, batch, desc_output, all_sels, compute_metric=False):
        num_batches = len(batch[DS.PC][0])
        all_flow_loss = []
        all_epe3d = []
        all_epe3d_full = []

        metric = PairwiseFlowMetric(compute_epe3d=True, compute_acc3d_outlier=False)
        for batch_idx in range(num_batches):
            cur_pc0, cur_pc1 = batch[DS.PC][0][batch_idx], batch[DS.PC][1][batch_idx]
            cur_sel0, cur_sel1 = all_sels[batch_idx * 2 + 0], all_sels[batch_idx * 2 + 1]
            cur_gt0, cur_gt1 = batch[DS.FULL_FLOW][(0, 1)][batch_idx], batch[DS.FULL_FLOW][(1, 0)][batch_idx]
            cur_mask0, cur_mask1 = batch[DS.FULL_MASK][(0, 1)][batch_idx], batch[DS.FULL_MASK][(1, 0)][batch_idx]
            cur_feat0 = desc_output.features_at(batch_idx * 2 + 0)
            cur_feat1 = desc_output.features_at(batch_idx * 2 + 1)
            dist_mat = torch.cdist(cur_feat0, cur_feat1) / torch.maximum(
                torch.tensor(np.float32(self.hparams.td_min), device=self.device), self.td)
            cur_pd0 = torch.softmax(-dist_mat, dim=1) @ cur_pc1[cur_sel1] - cur_pc0[cur_sel0]
            cur_pd1 = torch.softmax(-dist_mat, dim=0).transpose(-1, -2) @ cur_pc0[cur_sel0] - cur_pc1[cur_sel1]
            if cur_gt0 is not None:
                flow_loss01 = torch.linalg.norm(cur_pd0 - cur_gt0[cur_sel0], dim=-1)[cur_mask0[cur_sel0]].mean()
                all_flow_loss.append(flow_loss01)
            if cur_gt1 is not None:
                flow_loss10 = torch.linalg.norm(cur_pd1 - cur_gt1[cur_sel1], dim=-1)[cur_mask1[cur_sel1]].mean()
                all_flow_loss.append(flow_loss10)

            if compute_metric:
                with torch.no_grad():
                    if cur_gt0 is not None:
                        pd_full_flow01 = propagate_features(cur_pc0[cur_sel0], cur_pc0, cur_pd0, batched=False)
                        epe3d01 = metric.evaluate(cur_gt0, pd_full_flow01, cur_mask0)['epe3d']
                        epe3d01_full = metric.evaluate(cur_gt0, pd_full_flow01)['epe3d']
                        all_epe3d.append(epe3d01.item())
                        all_epe3d_full.append(epe3d01_full.item())
                    if cur_gt1 is not None:
                        pd_full_flow10 = propagate_features(cur_pc1[cur_sel1], cur_pc1, cur_pd1, batched=False)
                        epe3d10 = metric.evaluate(cur_gt1, pd_full_flow10, cur_mask1)['epe3d']
                        epe3d10_full = metric.evaluate(cur_gt1, pd_full_flow10)['epe3d']
                        all_epe3d.append(epe3d10.item())
                        all_epe3d_full.append(epe3d10_full.item())

        flow_loss = sum(all_flow_loss) / len(all_flow_loss)
        if compute_metric:
            metric_dict = {'epe3d': np.mean(all_epe3d), 'epe3d_full': np.mean(all_epe3d_full)}
        else:
            metric_dict = {}
        return {'flow': flow_loss}, metric_dict

    def training_step(self, batch, batch_idx):
        desc_output, all_sels = self(batch)
        loss_dict, metric_dict = self.compute_loss(batch, desc_output, all_sels, compute_metric=False)
        for metric_name, metric_val in metric_dict.items():
            self.log(f'train_loss/{metric_name}', metric_val)
        for loss_name, loss_val in loss_dict.items():
            self.log(f'train_loss/{loss_name}', loss_val)
        loss_sum = sum([t for t in loss_dict.values()])
        self.log('train_loss/sum', loss_sum)
        return loss_sum

    def validation_step(self, batch, batch_idx):
        desc_output, all_sels = self(batch)
        loss_dict, metric_dict = self.compute_loss(batch, desc_output, all_sels, compute_metric=True)
        for metric_name, metric_val in metric_dict.items():
            self.log(f'val_loss/{metric_name}', metric_val)
        for loss_name, loss_val in loss_dict.items():
            self.log(f'val_loss/{loss_name}', loss_val)
        loss_sum = sum([t for t in loss_dict.values()])
        self.log('val_loss', loss_sum)
        return loss_sum

    def get_dataset_spec(self):
        return [DS.FILENAME, DS.QUANTIZED_COORDS, DS.PC, DS.FULL_FLOW, DS.FULL_MASK]

    def train_dataloader(self):
        train_set = FlowDataset(**self.hparams.train_kwargs, spec=self.get_dataset_spec(),
                                hparams=self.hparams, augmentor=DataAugmentor(self.hparams.train_augmentation))
        torch.manual_seed(0)        # Ensure shuffle is consistent.
        return DataLoader(train_set, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=4, collate_fn=list_collate)

    def val_dataloader(self):
        val_set = FlowDataset(**self.hparams.val_kwargs, spec=self.get_dataset_spec(), hparams=self.hparams)
        return DataLoader(val_set, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=4, collate_fn=list_collate)
