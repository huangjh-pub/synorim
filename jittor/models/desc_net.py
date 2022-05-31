import jittor

from dataset import DatasetSpec as DS, FlowDataset

# Following won't work in jittor.
# from backbones.pointconv import PointConv as Backbone
from backbones.pointnet2 import PN2BackboneLarge as Backbone
from models.base_model import BaseModel
import numpy as np
from utils.misc import cdist


class Model(BaseModel):
    """
    This model trains the descriptor network. This is the 1st stage of training.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.backbone_args = self.hparams.backbone_args
        self.backbone = Backbone(out_channels=self.backbone_args.out_channels)
        self.td = jittor.array(np.float32(self.hparams.td_init))

    def execute(self, batch):
        B, K, N, _ = batch[DS.PC].shape
        aggr_pc = batch[DS.PC].view(-1, N, 3)
        feat = self.backbone(aggr_pc, aggr_pc).view(B, K, 8192, -1)
        feat = feat / jittor.norm(feat, dim=-1, keepdim=True)
        return feat

    def compute_loss(self, batch, desc_feat):
        cur_feat0 = desc_feat[:, 0]
        cur_feat1 = desc_feat[:, 1]
        cur_pc0, cur_pc1 = batch[DS.PC][:, 0], batch[DS.PC][:, 1]
        cur_gt0, cur_gt1 = batch[DS.FULL_FLOW][:, 0], batch[DS.FULL_FLOW][:, 1]

        dist_mat = cdist(cur_feat0, cur_feat1) / jittor.maximum(
            jittor.array(np.float32(self.hparams.td_min)), self.td)

        cur_pd0 = jittor.nn.softmax(-dist_mat, dim=2) @ cur_pc1 - cur_pc0
        cur_pd1 = jittor.nn.softmax(-dist_mat, dim=1).transpose(-1, -2) @ cur_pc0 - cur_pc1
        flow_loss01 = jittor.norm(cur_pd0 - cur_gt0, dim=-1).mean()
        flow_loss10 = jittor.norm(cur_pd1 - cur_gt1, dim=-1).mean()

        return {'flow': flow_loss01 + flow_loss10}

    def training_step(self, batch, batch_idx):
        desc_feat = self(batch)
        loss_dict = self.compute_loss(batch, desc_feat)
        loss_sum = sum([t for t in loss_dict.values()])
        return loss_sum

    def validation_step(self, batch, batch_idx):
        desc_feat = self(batch)
        loss_dict = self.compute_loss(batch, desc_feat)
        loss_sum = sum([t for t in loss_dict.values()])
        return loss_sum

    def train_dataloader(self):
        train_set = FlowDataset(**self.hparams.train_kwargs,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                num_workers=4,
                                spec=[DS.PC, DS.FULL_FLOW, DS.FULL_MASK])
        return train_set

    def val_dataloader(self):
        val_set = FlowDataset(**self.hparams.val_kwargs,
                              batch_size=self.hparams.batch_size,
                              shuffle=False,
                              num_workers=4,
                              spec=[DS.PC, DS.FULL_FLOW, DS.FULL_MASK])
        return val_set
