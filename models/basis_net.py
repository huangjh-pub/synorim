import random
from collections import defaultdict

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader

from dataset.base import DatasetSpec as DS, list_collate
from dataset.flow_dataset import FlowDataset, DataAugmentor
from metric import PairwiseFlowMetric

from models.spconv import ResUNet
from models.base_model import BaseModel
import numpy as np
from utils.point import propagate_features


class HuberKernel:
    """
    Huber loss influence function: Equ.(S.9), with the square root being pre-applied.
    """
    def __init__(self, robust_k):
        self.robust_k = robust_k

    def apply(self, b_mat, a_mat, c_cur):
        if c_cur is None:
            return torch.ones((a_mat.size(0), ), device=a_mat.device)
        fit_res = torch.norm(b_mat - torch.matmul(a_mat, c_cur), dim=-1)
        sigma = self.robust_k
        w_func = torch.where(fit_res < sigma, torch.ones_like(fit_res), sigma / fit_res)
        return torch.sqrt(w_func)


class Model(BaseModel):
    """
    This model trains the basis&refine network. This is the 2nd stage of training.
    """
    def __init__(self, hparams):
        super().__init__(hparams)

        self.backbone = ResUNet(self.hparams.backbone_args,
                                in_channels=3,
                                out_channels=self.hparams.n_basis,
                                normalize_feature=False,
                                conv1_kernel_size=3)
        self.refine = ResUNet(self.hparams.refine_args,
                              in_channels=9,
                              out_channels=3,
                              normalize_feature=False,
                              conv1_kernel_size=3)
        self.t = Parameter(torch.tensor(np.float32(self.hparams.t_init)), requires_grad=True)

        self.desc_net = self.load_module(self.hparams.desc_checkpoint)
        assert self.desc_net.hparams.voxel_size == self.hparams.voxel_size, "Voxel size of two models must match!"

    def forward(self, batch):
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

        self.desc_net.eval()
        with torch.no_grad():
            s_desc_output = self.desc_net.backbone(sinput)
        return [soutput, s_desc_output], all_sels

    @staticmethod
    def align_basis_via_gt(basis0, basis1, pc0, pc1, flow01, mask0):
        """
        Align basis0 and basis1 via ground-truth mapping
            Gt flow is always consistent, so we only need one direction.
        """
        warpped_pc0 = pc0 + flow01
        dist_mat = torch.cdist(warpped_pc0, pc1)       # (B, M, M)
        min_idx = torch.argmin(dist_mat, dim=-1)     # (B, M), (B, M)
        id0_mask = mask0
        id1_mask = min_idx[id0_mask]
        return basis0[id0_mask], basis1[id1_mask]

    def align_basis_via_pd_test(self, basis0, basis1, desc0, desc1, thres=0.3):
        """
        Align basis0 (N_k, M) and basis1 (N_l, M) using descriptor search.
        This function is used for test-time purpose with multiple inputs, not during training.
        """
        # Cross-check + distance prune.
        dist_mat = torch.cdist(desc0, desc1)
        min_dist, min_idx = torch.min(dist_mat, dim=-1)
        _, min_idx2 = torch.min(dist_mat, dim=-2)
        mutual_mask = min_idx2[min_idx] == torch.arange(min_idx.size(0), device=self.device)
        full_mask = torch.logical_and(mutual_mask, min_dist < thres)
        source_inds = torch.where(full_mask)[0]
        target_inds = min_idx[source_inds]
        return basis0[source_inds], basis1[target_inds]

    def align_basis_via_pd_train(self, basis0, basis1, desc0, desc1, force_output=False):
        """
        Align basis0 (N_k, M) and basis1 (N_l, M) using descriptor search.
        This function is used in training time.
        """
        dist_mat = torch.cdist(desc0, desc1)
        min_dist, min_idx = torch.min(dist_mat, dim=-1)
        _, min_idx2 = torch.min(dist_mat, dim=-2)
        mutual_mask = min_idx2[min_idx] == torch.arange(min_idx.size(0), device=self.device)
        # Relaxed for better stability.
        full_mask = torch.logical_and(mutual_mask, min_dist < 0.5)
        source_inds = torch.where(full_mask)[0]
        target_inds = min_idx[source_inds]
        if source_inds.size(0) < self.hparams.n_match_th:
            if force_output:
                # Even more relaxed if 0.5 still not works.
                source_inds = torch.where(min_dist < 0.8)[0]
                target_inds = min_idx[source_inds]
                return basis0[source_inds], basis1[target_inds]
            # Indicates matching failure...
            return None, None
        return basis0[source_inds], basis1[target_inds]

    def align_basis(self, basis0, basis1, desc0, desc1, pc0, pc1, flow01, mask0):
        """
        Random choose different strategies during training.
        """
        if random.random() < self.hparams.gt_align_prob and self.hparams.is_training and flow01 is not None:
            return self.align_basis_via_gt(basis0, basis1, pc0, pc1, flow01, mask0)
        else:
            # For validation we never use gt.
            phi_0, phi_1 = self.align_basis_via_pd_train(
                basis0, basis1, desc0, desc1, force_output=flow01 is None)
            # ... but if prediction fails, we have no other choices.
            if phi_0 is None:
                return self.align_basis_via_gt(basis0, basis1, pc0, pc1, flow01, mask0)
            return phi_0, phi_1

    def get_robust_kernel(self):
        if self.hparams.robust_kernel is not None:
            if self.hparams.robust_kernel.type == "huber":
                return HuberKernel(self.hparams.robust_kernel.robust_k), \
                       self.hparams.robust_kernel.robust_iter
            else:
                raise NotImplementedError
        return None, 1

    @staticmethod
    def solve_optimal_maps(phi_i, phi_j, robust_kernel=None, k_i=None, k_j=None, sqrt_mu=1.0,
                           c_init=None, num_iter=1):
        """
        This solves for the optimal FMap using the algorithm described in Alg.S1
        :param phi_i: phi_k^(kl)
        :param phi_j: phi_l^(kl)
        :param robust_kernel: robust kernel class
        :param k_i: H_k
        :param k_j: H_l
        :param sqrt_mu: relative weights of cycle term and data term. We use 1.
        :param c_init: initial maps. If not provided, it is assumed to be identity.
        :param num_iter: controls the IRLS iterations, if this is put under the synchronization, then 1 iter is enough.
        :return:
        """
        assert num_iter == 1 or robust_kernel is not None, "You do not have to iterate if there's no robust norm."

        if phi_i.size(0) < 5:
            # A well-posed solution is not feasible.
            # Leave initialization unchanged.
            if c_init is None:
                return torch.eye(phi_i.size(1), device=phi_i.device)
            return c_init

        c_current = c_init
        for iter_i in range(num_iter):
            # If there is robust kernel, then compute weight for this iteration.
            if robust_kernel is not None:
                sw = robust_kernel.apply(phi_j, phi_i, c_current)
                s_phi, s_ita = phi_j * sw.unsqueeze(-1), phi_i * sw.unsqueeze(-1)
            else:
                s_phi, s_ita = phi_j, phi_i

            if k_i is None or k_j is None or sqrt_mu == 0.0:
                # No kron or vec needed because without mu the expression can be simplified.
                # On GPU, inverse(ATA) is way faster -- however, it is not accurate which makes the algorithm diverge
                c_current = torch.pinverse(s_ita, rcond=1e-4) @ s_phi
            else:
                s_A = k_i * sqrt_mu
                s_B = k_j * sqrt_mu

                mI = torch.eye(s_phi.size(1), device=phi_i.device)
                nI = torch.eye(s_ita.size(1), device=phi_j.device)

                # Gamma_kl
                left_mat = torch.kron(torch.matmul(s_ita.transpose(-1, -2), s_ita), nI) + \
                           torch.kron(mI, torch.matmul(s_B, s_B.transpose(-1, -2)))
                # b_kl
                right_mat = (torch.matmul(s_ita.transpose(-1, -2), s_phi) +
                             torch.matmul(s_A, s_B.transpose(-1, -2))).view(-1)

                # --- cpu version for solve takes 80% less time.
                # c_current = torch.linalg.solve(left_mat, right_mat).view(s_phi.size(1), s_ita.size(1))
                c_current = torch.linalg.solve(left_mat.cpu(), right_mat.cpu()).view(
                    s_phi.size(1), s_ita.size(1)).to(left_mat.device)

        return c_current

    def compute_flow_from_maps(self, basis_i, basis_j, c_ij, sub_pc_i, sub_pc_j,
                               pcond_multiplier=None, sparse_coords=None):
        """
        Given the functional mapping, compute scene flow.
        :param basis_i: Bases of source point cloud (N_k, M)
        :param basis_j: Bases of target point cloud (N_l, M)
        :param c_ij: Functional maps (M, M)
        :param sub_pc_i: source point cloud (N_k, 3)
        :param sub_pc_j: target point cloud (N_l, 3)
        :param pcond_multiplier: \\Sigma V^T matrix used in preconditioning
        :param sparse_coords: Spconv coordinates for forwarding refinement network.
        :return: dict of (N_k, 3) flow vectors
        """
        basis_i_aligned = basis_i @ c_ij
        output_flow = {}

        # Compute F^f
        pc_j_center = torch.mean(sub_pc_j, dim=0, keepdim=True)
        inv_basis_j = torch.pinverse(basis_j, rcond=1e-4)
        basis_flow = basis_i_aligned @ inv_basis_j @ (sub_pc_j - pc_j_center) + pc_j_center - sub_pc_i
        output_flow['f'] = basis_flow

        # Compute F^n
        if pcond_multiplier is not None:
            basis_dist = torch.cdist(basis_i_aligned @ pcond_multiplier, basis_j @ pcond_multiplier)
        else:
            basis_dist = torch.cdist(basis_i_aligned, basis_j)

        soft_corr_mat = F.softmax(-basis_dist / torch.maximum(
            torch.tensor(np.float32(self.hparams.t_min), device=self.device), self.t), dim=-1)
        ot_flow = torch.matmul(soft_corr_mat, sub_pc_j) - sub_pc_i

        # Compute final F
        all_feats = [sub_pc_i, basis_flow, ot_flow]
        all_feats = [torch.cat(all_feats, dim=1)]
        coords_batch, feats_batch = ME.utils.sparse_collate([sparse_coords], all_feats, device=self.device)
        sinput = ME.SparseTensor(feats_batch, coordinates=coords_batch)
        soutput = self.refine(sinput)

        output_flow['final'] = soutput.features_at(0) + ot_flow
        return output_flow

    def compute_loss(self, batch, basis_output, all_sels, compute_metric=False):
        num_batches = len(batch[DS.PC][0])
        loss_dict = defaultdict(list)
        metric_dict = defaultdict(list)

        metric = PairwiseFlowMetric(compute_epe3d=True, compute_acc3d_outlier=False)
        robust_kernel, robust_iter = self.get_robust_kernel()

        for batch_idx in range(num_batches):
            cur_pc0, cur_pc1 = batch[DS.PC][0][batch_idx], batch[DS.PC][1][batch_idx]
            cur_sel0, cur_sel1 = all_sels[batch_idx * 2 + 0], all_sels[batch_idx * 2 + 1]
            cur_gt0, cur_gt1 = batch[DS.FULL_FLOW][(0, 1)][batch_idx], batch[DS.FULL_FLOW][(1, 0)][batch_idx]
            cur_mask0, cur_mask1 = batch[DS.FULL_MASK][(0, 1)][batch_idx], batch[DS.FULL_MASK][(1, 0)][batch_idx]

            cur_sub_pc0, cur_sub_pc1 = cur_pc0[cur_sel0], cur_pc1[cur_sel1]

            if cur_gt0 is not None:
                cur_sub_gt0, cur_sub_mask0 = cur_gt0[cur_sel0], cur_mask0[cur_sel0]
            else:
                cur_sub_gt0, cur_sub_mask0 = None, None

            if cur_gt1 is not None:
                cur_sub_gt1, cur_sub_mask1 = cur_gt1[cur_sel1], cur_mask1[cur_sel1]
            else:
                cur_sub_gt1, cur_sub_mask1 = None, None

            cur_basis0 = basis_output[0].features_at(batch_idx * 2 + 0)
            cur_basis1 = basis_output[0].features_at(batch_idx * 2 + 1)
            cur_desc0 = basis_output[1].features_at(batch_idx * 2 + 0)
            cur_desc1 = basis_output[1].features_at(batch_idx * 2 + 1)

            aligned_basis0, aligned_basis1 = self.align_basis(
                cur_basis0, cur_basis1, cur_desc0, cur_desc1, cur_sub_pc0, cur_sub_pc1, cur_sub_gt0, cur_sub_mask0)
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

            if cur_sub_gt0 is not None:
                loss_dict['flow'].append(torch.linalg.norm(flow01['final'] - cur_sub_gt0, dim=-1).mean())
            if cur_sub_gt1 is not None:
                loss_dict['flow'].append(torch.linalg.norm(flow10['final'] - cur_sub_gt1, dim=-1).mean())

            if self.hparams.flow_f_weight > 0.0:
                if cur_sub_gt0 is not None:
                    basis_err01 = torch.linalg.norm(flow01['f'] - cur_sub_gt0, dim=-1)
                    basis_err01 = basis_err01[basis_err01 < self.hparams.flow_f_max_bound]
                    if basis_err01.size(0) > 0:
                        loss_dict['flow_f'].append(self.hparams.flow_f_weight * basis_err01.mean())
                if cur_sub_gt1 is not None:
                    basis_err10 = torch.linalg.norm(flow10['f'] - cur_sub_gt1, dim=-1)
                    basis_err10 = basis_err10[basis_err10 < self.hparams.flow_f_max_bound]
                    if basis_err10.size(0) > 0:
                        loss_dict['flow_f'].append(self.hparams.flow_f_weight * basis_err10.mean())

            if self.hparams.smoothness_weight > 0.0:
                dist00 = torch.cdist(cur_sub_pc0, cur_sub_pc0)
                _, kidx0 = torch.topk(dist00, 10, dim=-1, largest=False, sorted=False)
                grouped_flow0 = flow01['final'][kidx0]  # (N, K, 3)
                smooth_loss0 = self.hparams.smoothness_weight * \
                                (((grouped_flow0 - flow01['final'].unsqueeze(1)) ** 2).sum(-1).sum(-1) / 9.0).mean()
                dist11 = torch.cdist(cur_sub_pc1, cur_sub_pc1)
                _, kidx1 = torch.topk(dist11, 10, dim=-1, largest=False, sorted=False)
                grouped_flow1 = flow10['final'][kidx1]  # (N, K, 3)
                smooth_loss1 = self.hparams.smoothness_weight * \
                                (((grouped_flow1 - flow10['final'].unsqueeze(1)) ** 2).sum(-1).sum(-1) / 9.0).mean()
                loss_dict['smooth'].append(smooth_loss0)
                loss_dict['smooth'].append(smooth_loss1)

            if compute_metric:
                with torch.no_grad():
                    if cur_gt0 is not None:
                        pd_full_flow01 = propagate_features(cur_pc0[cur_sel0], cur_pc0, flow01['final'], batched=False)
                        epe3d01 = metric.evaluate(cur_gt0, pd_full_flow01, cur_mask0)['epe3d']
                        epe3d01_full = metric.evaluate(cur_gt0, pd_full_flow01)['epe3d']
                        metric_dict[f'epe3d'].append(epe3d01.item())
                        metric_dict[f'epe3d-full'].append(epe3d01_full.item())
                    if cur_gt1 is not None:
                        pd_full_flow10 = propagate_features(cur_pc1[cur_sel1], cur_pc1, flow10['final'], batched=False)
                        epe3d10 = metric.evaluate(cur_gt1, pd_full_flow10, cur_mask1)['epe3d']
                        epe3d10_full = metric.evaluate(cur_gt1, pd_full_flow10)['epe3d']
                        metric_dict[f'epe3d'].append(epe3d10.item())
                        metric_dict[f'epe3d-full'].append(epe3d10_full.item())

        loss_dict = {k: sum(v) / len(v) for k, v in loss_dict.items()}
        if compute_metric:
            metric_dict = {k: np.mean(v) for k, v in metric_dict.items()}
        else:
            metric_dict = {}
        return loss_dict, metric_dict

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
        return [DS.QUANTIZED_COORDS, DS.PC, DS.FULL_FLOW, DS.FULL_MASK]

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
