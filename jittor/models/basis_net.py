import jittor
import random

from dataset import DatasetSpec as DS, FlowDataset

from backbones.pointnet2 import PN2BackboneLarge as Backbone
from backbones.pointnet2 import PN2BackboneSmall as BackboneSmall
from models.base_model import BaseModel
import numpy as np
from utils.misc import cdist, cdist_single
from collections import defaultdict


class HuberKernel:
    """
    Huber loss influence function: Equ.(S.9), with the square root being pre-applied.
    """
    def __init__(self, robust_k):
        self.robust_k = robust_k

    def apply(self, b_mat, a_mat, c_cur):
        if c_cur is None:
            return jittor.ones((a_mat.shape[0], ))
        fit_res = jittor.norm(b_mat - jittor.matmul(a_mat, c_cur), dim=-1)
        sigma = self.robust_k
        w_func = jittor.ones_like(fit_res)
        w_func[fit_res < sigma] = (sigma / fit_res)[fit_res < sigma]
        return jittor.sqrt(w_func)


class Model(BaseModel):
    """
    This model trains the basis&refine network. This is the 2nd stage of training.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.backbone = Backbone(out_channels=self.hparams.n_basis)
        self.refine = BackboneSmall(out_channels=3)
        self.t = jittor.array(np.float32(self.hparams.t_init))
        self.desc_net = self.load_module(self.hparams.desc_checkpoint)

    def execute(self, batch):
        B, K, N, _ = batch[DS.PC].shape
        aggr_pc = batch[DS.PC].view(-1, N, 3)
        basis = self.backbone(aggr_pc, aggr_pc).view(B, K, N, -1)

        self.desc_net.eval()
        with jittor.no_grad():
            desc = self.desc_net(batch)

        return basis, desc

    @staticmethod
    def align_basis_via_gt(basis0, basis1, pc0, pc1, flow01, mask0):
        warpped_pc0 = pc0 + flow01
        dist_mat = cdist_single(warpped_pc0, pc1)       # (B, M, M)
        min_idx, _ = jittor.argmin(dist_mat, dim=-1)     # (B, M), (B, M)
        id0_mask = mask0
        id1_mask = min_idx[id0_mask]
        return basis0[id0_mask], basis1[id1_mask]

    def align_basis_via_pd_test(self, basis0, basis1, desc0, desc1, thres=0.3):
        # Cross-check + distance prune.
        dist_mat = cdist_single(desc0, desc1)
        min_idx, min_dist = jittor.argmin(dist_mat, dim=1)
        min_idx2, _ = jittor.argmin(dist_mat, dim=0)
        mutual_mask = min_idx2[min_idx] == jittor.arange(min_idx.shape[0])
        full_mask = jittor.logical_and(mutual_mask, min_dist < thres)
        source_inds = jittor.where(full_mask)[0]
        target_inds = min_idx[source_inds]
        return basis0[source_inds], basis1[target_inds]

    def align_basis_via_pd_train(self, basis0, basis1, desc0, desc1, force_output=False):
        dist_mat = cdist_single(desc0, desc1)
        min_idx, min_dist = jittor.argmin(dist_mat, dim=1)
        min_idx2, _ = jittor.argmin(dist_mat, dim=0)
        mutual_mask = min_idx2[min_idx] == jittor.arange(min_idx.shape[0])
        # Relaxed for better stability.
        full_mask = jittor.logical_and(mutual_mask, min_dist < 0.5)
        source_inds = jittor.where(full_mask)[0]
        target_inds = min_idx[source_inds]
        if source_inds.size(0) < self.hparams.n_match_th:
            if force_output:
                # Even more relaxed if 0.5 still not works.
                source_inds = jittor.where(min_dist < 0.8)[0]
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
        assert num_iter == 1 or robust_kernel is not None, "You do not have to iterate if there's no robust norm."

        if phi_i.shape[0] < 5:
            # A well-posed solution is not feasible.
            # Leave initialization unchanged.
            if c_init is None:
                return jittor.init.eye(phi_i.shape[1])
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
                c_current = jittor.linalg.pinv(s_ita) @ s_phi
            else:
                s_A = k_i * sqrt_mu
                s_B = k_j * sqrt_mu

                mI = jittor.init.eye(s_phi.shape[1])
                nI = jittor.init.eye(s_ita.shape[1])

                s_ita, s_phi, s_B, s_A, mI, nI = \
                    s_ita.numpy(), s_phi.numpy(), s_B.numpy(), s_A.numpy(), mI.numpy(), nI.numpy()

                left_mat = np.kron(np.matmul(s_ita.transpose(-1, -2), s_ita), nI) + \
                           np.kron(mI, np.matmul(s_B, s_B.transpose(-1, -2)))
                right_mat = (np.matmul(s_ita.transpose(-1, -2), s_phi) +
                             np.matmul(s_A, s_B.transpose(-1, -2))).view(-1)
                c_current = np.linalg.solve(left_mat, right_mat).view(s_phi.shape[1], s_ita.shape[1])

                c_current = jittor.array(c_current)

        return c_current

    def compute_flow_from_maps(self, basis_i, basis_j, c_ij, sub_pc_i, sub_pc_j,
                               pcond_multiplier=None):
        basis_i_aligned = basis_i @ c_ij
        output_flow = {}

        # Compute F^f
        pc_j_center = jittor.mean(sub_pc_j, dim=0, keepdims=True)
        inv_basis_j = jittor.linalg.pinv(basis_j)
        basis_flow = basis_i_aligned @ inv_basis_j @ (sub_pc_j - pc_j_center) + pc_j_center - sub_pc_i
        output_flow['f'] = basis_flow

        # Compute F^n
        if pcond_multiplier is not None:
            basis_dist = cdist_single(basis_i_aligned @ pcond_multiplier, basis_j @ pcond_multiplier)
        else:
            basis_dist = cdist_single(basis_i_aligned, basis_j)

        soft_corr_mat = jittor.nn.softmax(-basis_dist / jittor.maximum(
            jittor.array(np.float32(self.hparams.t_min)), self.t), dim=-1)
        ot_flow = jittor.matmul(soft_corr_mat, sub_pc_j) - sub_pc_i

        # Compute final F
        all_feats = [sub_pc_i, basis_flow, ot_flow]
        all_feats = jittor.concat(all_feats, dim=1)
        delta_flow = self.refine(sub_pc_i.unsqueeze(0), all_feats.unsqueeze(0))

        output_flow['final'] = delta_flow[0].permute(1, 0) + ot_flow
        return output_flow

    def compute_loss(self, batch, basis_output, desc_output):
        num_batches = len(batch[DS.PC][0])
        loss_dict = defaultdict(list)

        robust_kernel, robust_iter = self.get_robust_kernel()

        for batch_idx in range(num_batches):
            cur_pc0, cur_pc1 = batch[DS.PC][0][batch_idx], batch[DS.PC][1][batch_idx]
            cur_gt0, cur_gt1 = batch[DS.FULL_FLOW][(0, 1)][batch_idx], batch[DS.FULL_FLOW][(1, 0)][batch_idx]
            cur_mask0, cur_mask1 = batch[DS.FULL_MASK][(0, 1)][batch_idx], batch[DS.FULL_MASK][(1, 0)][batch_idx]

            cur_basis0 = basis_output[batch_idx, 0]
            cur_basis1 = basis_output[batch_idx, 1]
            cur_desc0 = desc_output[batch_idx, 0]
            cur_desc1 = desc_output[batch_idx, 1]

            aligned_basis0, aligned_basis1 = self.align_basis(
                cur_basis0, cur_basis1, cur_desc0, cur_desc1, cur_pc0, cur_pc1, cur_gt0, cur_mask0)
            c01 = self.solve_optimal_maps(aligned_basis0, aligned_basis1,
                                          robust_kernel=robust_kernel, num_iter=robust_iter)
            c10 = self.solve_optimal_maps(aligned_basis1, aligned_basis0,
                                          robust_kernel=robust_kernel, num_iter=robust_iter)

            if self.hparams.ctc_weight:
                ctc_err = c01 @ c10 - jittor.init.eye(c01.shape[0])
                loss_dict['ctc'].append(self.hparams.ctc_weight * jittor.sum(ctc_err ** 2))

            flow01 = self.compute_flow_from_maps(cur_basis0, cur_basis1, c01, cur_pc0, cur_pc1)
            flow10 = self.compute_flow_from_maps(cur_basis1, cur_basis0, c10, cur_pc1, cur_pc0)

            loss_dict['flow'].append(jittor.norm(flow01['final'] - cur_gt0, dim=-1).mean())
            loss_dict['flow'].append(jittor.norm(flow10['final'] - cur_gt1, dim=-1).mean())

        loss_dict = {k: sum(v) / len(v) for k, v in loss_dict.items()}
        return loss_dict

    def training_step(self, batch, batch_idx):
        basis_output, desc_output = self(batch)
        loss_dict = self.compute_loss(batch, basis_output, desc_output)
        loss_sum = sum([t for t in loss_dict.values()])
        return loss_sum

    def validation_step(self, batch, batch_idx):
        basis_output, desc_output = self(batch)
        loss_dict = self.compute_loss(batch, basis_output, desc_output)
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
