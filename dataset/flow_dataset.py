import json
from pathlib import Path

import numpy as np

from dataset.base import RandomSafeDataset, DatasetSpec

import MinkowskiEngine as ME
from pyquaternion.quaternion import Quaternion


class DataAugmentor:
    """
    Will apply data augmentation to pairwise point clouds, by applying random transformations
    to the point clouds (or one of them), or adding noise.
    """
    def __init__(self, args):
        self.args = args
        self.together_args = self.args.together
        self.pc2_args = self.args.pc2

    def process(self, data_dict: dict, rng: np.random.RandomState):
        assert DatasetSpec.PC in data_dict.keys()
        assert DatasetSpec.FULL_FLOW in data_dict.keys()

        pcs = data_dict[DatasetSpec.PC]
        assert len(pcs) == 2

        pc1, pc2 = pcs[0], pcs[1]
        if data_dict[DatasetSpec.FULL_FLOW][(1, 0)] is not None:
            pc1_virtual = pc2 + data_dict[DatasetSpec.FULL_FLOW][(1, 0)]
        else:
            pc1_virtual = None
        if data_dict[DatasetSpec.FULL_FLOW][(0, 1)] is not None:
            pc2_virtual = pc1 + data_dict[DatasetSpec.FULL_FLOW][(0, 1)]
        else:
            pc2_virtual = None

        if self.args.centralize:
            pc1_center = np.mean(pc1, axis=0, keepdims=True)
            pc2_center = np.mean(pc2, axis=0, keepdims=True)
            pc1 -= pc1_center
            pc2 -= pc2_center
            if pc2_virtual is not None:
                pc2_virtual -= pc2_center

        if self.together_args.enabled:
            scale = np.diag(rng.uniform(self.together_args.scale_low,
                                        self.together_args.scale_high, 3).astype(np.float32))
            angle = rng.uniform(-self.together_args.degree_range, self.together_args.degree_range) / 180.0 * np.pi
            rot_matrix = np.array([
                [np.cos(angle), 0., np.sin(angle)],
                [0., 1., 0.],
                [-np.sin(angle), 0., np.cos(angle)]
            ], dtype=np.float32)
            matrix = scale.dot(rot_matrix.T)
            bias = rng.uniform(-self.together_args.shift_range,
                               self.together_args.shift_range, (1, 3)).astype(np.float32) + \
                   np.clip(self.together_args.jitter_sigma * rng.randn(pc1.shape[0], 3),
                           -self.together_args.jitter_clip, self.together_args.jitter_clip).astype(np.float32)

            pc1 = pc1.dot(matrix) + bias
            pc2 = pc2.dot(matrix) + bias
            if pc2_virtual is not None:
                pc2_virtual = pc2_virtual.dot(matrix) + bias

        if self.pc2_args.enabled:
            angle2 = rng.uniform(-self.pc2_args.degree_range, self.pc2_args.degree_range) / 180.0 * np.pi
            if self.pc2_args.dof == 'y':
                rot_axis = np.array([0.0, 1.0, 0.0])
            elif self.pc2_args.dof == 'full':
                rot_axis = rng.randn(3)
                rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1.0e-6)
            else:
                raise NotImplementedError
            matrix2 = Quaternion(axis=rot_axis, radians=angle2).rotation_matrix.astype(np.float32)
            shifts2 = rng.uniform(-self.pc2_args.shift_range,
                                  self.pc2_args.shift_range, (1, 3)).astype(np.float32)
            jitter2 = np.clip(self.pc2_args.jitter_sigma * rng.randn(pc1.shape[0], 3),
                              -self.pc2_args.jitter_clip, self.pc2_args.jitter_clip).astype(np.float32)

            pc2 = pc2.dot(matrix2) + shifts2 + jitter2
            if pc2_virtual is not None:
                pc2_virtual = pc2_virtual.dot(matrix2) + shifts2

        data_dict[DatasetSpec.PC] = [pc1, pc2]
        if pc2_virtual is not None:
            data_dict[DatasetSpec.FULL_FLOW][(0, 1)] = pc2_virtual - pc1

        if pc1_virtual is not None:
            data_dict[DatasetSpec.FULL_FLOW][(1, 0)] = pc1_virtual - pc2


class FlowDataset(RandomSafeDataset):
    def __init__(self, base_folder: str, spec: list, sub_frames: list, split: str,
                 augmentor: DataAugmentor = None, random_seed: [int, str] = 0, hparams=None):
        """
        :param base_folder: path to the dataset
        :param spec: a list of DataSpec enums, specify which kinds of data are needed.
        :param sub_frames: a list of frame subsets, each subset should be a list containing frame indices.
        :param split: <file>:<split>, will load the split key in file.json.
        :param augmentor: Data augmentation
        :param random_seed: int or "fixed" string.
        :param hparams: oconf hyper parameters needed.
        """
        if isinstance(random_seed, str):
            super().__init__(0, True)
        else:
            super().__init__(random_seed, False)
        self.base_folder = Path(base_folder)
        self.full_split = split
        meta_file, self.split = split.split(':')
        with (self.base_folder / f"{meta_file}.json").open() as f:
            self.meta = json.load(f)
        data_defs = self.meta[self.split]
        self.data_files = [t[0] for t in data_defs]
        self.sub_frames = sub_frames
        self.spec = spec
        self.hparams = hparams
        self.augmentor = augmentor

    def __len__(self):
        return len(self.data_files) * len(self.sub_frames)

    def __getitem__(self, data_id):
        rng = self.get_rng(data_id)
        try:
            return self._get_item(data_id, rng)
        except EOFError:
            # For DD dataset, the sampled pair may not have any associated flow during training,
            # in that case this data is skipped.
            return self.__getitem__(rng.randint(0, len(self) - 1))

    def _get_item(self, data_id, rng):
        idx = data_id // len(self.sub_frames)
        view_sel_idx = self.sub_frames[data_id % len(self.sub_frames)]

        data_path = self.base_folder / "data" / self.data_files[idx]
        datum = np.load(data_path, allow_pickle=True)

        final_pcs = []
        for vid in view_sel_idx:
            final_pcs.append(datum['pcs'][vid])

        ret_vals = {}
        if DatasetSpec.FILENAME in self.spec:
            ret_vals[DatasetSpec.FILENAME] = str(data_path)

        if DatasetSpec.PC in self.spec:
            final_pcs = [t.astype(np.float32) for t in final_pcs]
            ret_vals[DatasetSpec.PC] = final_pcs

        if DatasetSpec.FULL_FLOW in self.spec:
            full_flows = {}
            has_valid_flow = False
            for fidx, view_i in enumerate(view_sel_idx):
                for fjdx, view_j in enumerate(view_sel_idx):
                    if view_i == view_j:
                        continue
                    cur_flow = datum['flows'][view_i][view_j]
                    if cur_flow is None:
                        final_flows = None
                    else:
                        final_flows = np.nan_to_num(cur_flow.astype(np.float32))
                        has_valid_flow = True
                    full_flows[(fidx, fjdx)] = final_flows
            ret_vals[DatasetSpec.FULL_FLOW] = full_flows

            if not has_valid_flow:
                raise EOFError

        if DatasetSpec.FULL_MASK in self.spec:
            full_masks = {}
            for fidx, view_i in enumerate(view_sel_idx):
                for fjdx, view_j in enumerate(view_sel_idx):
                    if view_i == view_j:
                        continue
                    if 'flow_masks' not in datum.keys():
                        full_masks[(fidx, fjdx)] = np.ones(
                            (ret_vals[DatasetSpec.FULL_FLOW][(fidx, fjdx)].shape[0], ), dtype=bool)
                        continue
                    cur_occ = datum['flow_masks'][view_i][view_j]
                    full_masks[(fidx, fjdx)] = cur_occ
            ret_vals[DatasetSpec.FULL_MASK] = full_masks

        if self.augmentor is not None:
            self.augmentor.process(ret_vals, rng)

        if DatasetSpec.QUANTIZED_COORDS in self.spec:
            quan_coords = []
            for cur_pc in ret_vals[DatasetSpec.PC]:
                coords = np.floor(cur_pc / self.hparams.voxel_size)
                inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)
                quan_coords.append((coords[inds], inds))
            ret_vals[DatasetSpec.QUANTIZED_COORDS] = quan_coords

        return ret_vals
