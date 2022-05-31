from collections import defaultdict

import jittor

from dataset import DatasetSpec as DS, FlowDataset
from metric import PairwiseFlowMetric

from models.base_model import BaseModel
import numpy as np


class Model(BaseModel):
    """
    This model runs the full test of our model, taking multiple point clouds as input.
    """
    def __init__(self, hparams):
        super().__init__(hparams)
        self.basis_net = self.load_module(self.hparams.basis_checkpoint)
        self.desc_net = self.basis_net.desc_net

    def execute(self, batch):
        s_bases, s_descs = self.basis_net(batch)
        return s_bases, s_descs

    def optimize_map(self, old_c_dict, ca_dict, cb_dict):
        all_keys = sorted(list(old_c_dict.keys()))
        num_frames = max(*[max(t) for t in all_keys]) + 1
        view_sizes = {}
        for fid in range(num_frames - 1):
            view_sizes[fid] = old_c_dict[(fid, fid + 1)].size(0)
            view_sizes[fid + 1] = old_c_dict[(fid, fid + 1)].size(1)
        var_sizes_T = np.cumsum([0] + [view_sizes[i] for i in range(num_frames)])
        num_universe = max(view_sizes.values()) - self.hparams.num_v_sub
        consistent_weight = self.hparams.cycle_weight
        C_star = {k: v for k, v in old_c_dict.items()}
        C_init_scale = {k: jittor.norm(v.flatten()) for k, v in C_star.items()}
        robust_kernel, _ = self.basis_net.get_robust_kernel()
        for iter_i in range(self.hparams.sync_iter):
            h_rows = []
            for fid_i in range(num_frames):
                h_cols = []
                for fid_j in range(num_frames):
                    sum_matrices = [jittor.zeros((view_sizes[fid_i], view_sizes[fid_j]))]
                    if fid_i < fid_j:
                        if (fid_i, fid_j) in all_keys:
                            sum_matrices.append(-C_star[(fid_i, fid_j)])
                        if (fid_j, fid_i) in all_keys:
                            sum_matrices.append(-C_star[(fid_j, fid_i)].transpose(-1, -2))
                    elif fid_i == fid_j:
                        for fid_k in range(num_frames):
                            if (fid_i, fid_k) in all_keys:
                                sum_matrices.append(jittor.init.eye(view_sizes[fid_i]))
                            if (fid_k, fid_i) in all_keys:
                                X_ji = C_star[(fid_k, fid_i)]
                                sum_matrices.append(X_ji.transpose(-1, -2) @ X_ji)
                    h_cols.append(sum(sum_matrices))
                h_rows.append(jittor.concat(h_cols, dim=-1))
            full_h_matrix = jittor.concat(h_rows, dim=0)
            _, h_star = jittor.linalg.eigh(full_h_matrix)
            h_star = h_star[..., :num_universe]
            change_scales = []
            for (mid, nid) in all_keys:
                C_ij_star = self.basis_net.solve_optimal_maps(
                    ca_dict[(mid, nid)], cb_dict[(mid, nid)],
                    robust_kernel=robust_kernel,
                    k_i=h_star[var_sizes_T[mid]: var_sizes_T[mid + 1]],
                    k_j=h_star[var_sizes_T[nid]: var_sizes_T[nid + 1]],
                    sqrt_mu=np.sqrt(consistent_weight),
                    c_init=C_star[(mid, nid)],
                )
                change_scale = jittor.norm((C_ij_star - C_star[(mid, nid)]).flatten()) / C_init_scale[(mid, nid)]
                change_scales.append(change_scale.item())
                C_star[(mid, nid)] = C_ij_star
            rel_change = np.mean(change_scales)
            if rel_change < self.hparams.sync_converge_rel:
                break
        return C_star

    def test_step(self, batch, batch_idx):
        # Forward descriptor and basis networks.
        basis_output, desc_output = self(batch)
        basis_output = basis_output[0]
        desc_output = desc_output[0]

        # Generate indices pairs and obtain data from batch.
        num_views = 4
        iters_ij, iters_upper_ij = [], []

        for view_i in range(num_views):
            for view_j in range(num_views):
                if view_i == view_j:
                    continue
                if view_i < view_j:
                    iters_upper_ij.append((view_i, view_j))
                iters_ij.append((view_i, view_j))

        # sub_pc is a subset of the full point cloud, selected by ME quantization
        full_pc = {}
        for view_i in range(num_views):
            full_pc[view_i] = batch[DS.PC][0, view_i]

        # Compute pairwise feature descriptors phi_k^kl, phi_l^kl for each pair
        phi_i_all, phi_j_all = {}, {}
        for (view_i, view_j) in iters_upper_ij:
            phi_i_all[(view_i, view_j)], phi_j_all[(view_i, view_j)] = self.basis_net.align_basis_via_pd_test(
                basis_output[view_i], basis_output[view_j],
                desc_output[view_i], desc_output[view_j],
                thres=self.hparams.gpd_thres
            )
            phi_i_all[(view_j, view_i)], phi_j_all[(view_j, view_i)] = \
                phi_j_all[(view_i, view_j)], phi_i_all[(view_i, view_j)]

        # Computes initial maps.
        maps_init = {}
        robust_kernel, robust_iter = self.basis_net.get_robust_kernel()
        for (view_i, view_j) in iters_ij:
            maps_init[(view_i, view_j)] = self.basis_net.solve_optimal_maps(
                phi_i_all[(view_i, view_j)], phi_j_all[(view_i, view_j)],
                robust_kernel=robust_kernel, num_iter=robust_iter
            )

        # Optimize maps via synchronization (Sec.5).
        maps_optimized = self.optimize_map(maps_init, phi_i_all, phi_j_all)

        # Generate flow.
        final_flows = {}
        for (view_i, view_j) in iters_ij:
            final_flows[(view_i, view_j)] = self.basis_net.compute_flow_from_maps(
                basis_output[view_i], basis_output[view_j], maps_optimized[(view_i, view_j)],
                full_pc[view_i], full_pc[view_j]
            )['final']

        # Measure errors.
        error = self.evaluate_flow_error(batch, final_flows)

        return final_flows, error

    @staticmethod
    def evaluate_flow_error(batch, pd_flows):
        eval_pairs = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3),
                      (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]
        err_acc_dict = defaultdict(list)

        for (view_i, view_j) in eval_pairs:
            gt_flow_ij = batch[DS.FULL_FLOW][0, view_i * 3 + view_j]
            gt_mask_ij = batch[DS.FULL_MASK][0, view_i * 3 + view_j]

            # If ground-truth does not exist for this pair, then ignore it.
            if gt_flow_ij is None:
                continue

            err_dict = PairwiseFlowMetric(compute_epe3d=True, compute_acc3d_outlier=True).evaluate(
                gt_flow_ij, pd_flows[(view_i, view_j)], valid_mask=gt_mask_ij)
            err_full_dict = PairwiseFlowMetric(compute_epe3d=True, compute_acc3d_outlier=True).evaluate(
                gt_flow_ij, pd_flows[(view_i, view_j)])

            err_dict = {k: v.item() for k, v in err_dict.items()}
            err_full_dict = {k: v.item() for k, v in err_full_dict.items()}

            # Put all metrics into list
            for err_name in err_dict.keys():
                err_acc_dict[err_name].append(err_dict[err_name])
                err_acc_dict[err_name + "-full"].append(err_full_dict[err_name])

        err_acc_final_dict = {}
        for mkey, marray in err_acc_dict.items():
            err_acc_final_dict[f"{mkey}-avg"] = np.mean(marray)
            err_acc_final_dict[f"{mkey}-std"] = np.std(marray)

        return err_acc_final_dict

    def test_dataloader(self):
        test_set = FlowDataset(**self.hparams.test_kwargs, spec=[
            DS.PC, DS.FULL_FLOW, DS.FULL_MASK], batch_size=1, shuffle=False, num_workers=4)
        return test_set
