import json
import jittor
from pathlib import Path
import numpy as np


class DatasetSpec:
    PC = 200
    FULL_FLOW = 300
    FULL_MASK = 400


class FlowDataset(jittor.dataset.Dataset):
    def __init__(self, batch_size, shuffle, num_workers,
                 base_folder: str, spec: list, sub_frames: list, split: str):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.base_folder = Path(base_folder)
        self.full_split = split
        meta_file, self.split = split.split(':')
        with (self.base_folder / f"{meta_file}.json").open() as f:
            self.meta = json.load(f)
        data_defs = self.meta[self.split]
        self.data_files = [t[0] for t in data_defs]
        self.sub_frames = list(sub_frames)
        self.spec = spec

        self.total_len = len(self.data_files) * len(self.sub_frames)

    def __getitem__(self, data_id):
        idx = data_id // len(self.sub_frames)
        view_sel_idx = self.sub_frames[data_id % len(self.sub_frames)]

        data_path = self.base_folder / "data" / self.data_files[idx]
        datum = np.load(data_path, allow_pickle=True)

        final_pcs = []
        for vid in view_sel_idx:
            final_pcs.append(datum['pcs'][vid])

        ret_vals = {}

        if DatasetSpec.PC in self.spec:
            final_pcs = [t.astype(np.float32) for t in final_pcs]
            ret_vals[DatasetSpec.PC] = np.stack(final_pcs, axis=0)

        if DatasetSpec.FULL_FLOW in self.spec:
            full_flows = []
            for fidx, view_i in enumerate(view_sel_idx):
                for fjdx, view_j in enumerate(view_sel_idx):
                    if view_i == view_j:
                        continue
                    cur_flow = datum['flows'][view_i][view_j]
                    final_flows = np.nan_to_num(cur_flow.astype(np.float32))
                    full_flows.append(final_flows)
            ret_vals[DatasetSpec.FULL_FLOW] = np.stack(full_flows, axis=0)

        if DatasetSpec.FULL_MASK in self.spec:
            full_masks = []
            for fidx, view_i in enumerate(view_sel_idx):
                for fjdx, view_j in enumerate(view_sel_idx):
                    if view_i == view_j:
                        continue
                    cur_occ = datum['flow_masks'][view_i][view_j]
                    full_masks.append(cur_occ)
            ret_vals[DatasetSpec.FULL_MASK] = np.stack(full_masks, axis=0)

        return ret_vals
