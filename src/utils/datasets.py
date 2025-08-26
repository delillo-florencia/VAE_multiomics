import h5py
import torch
import joblib
from torch.utils.data import Dataset


class ExpressionDataset(Dataset):
    def __init__(self, h5_path: str, sample_ids_list: list[str], scaler_path: str = None):
        self.h5 = h5py.File(h5_path, 'r')
        all_ids = [sid.decode('utf-8') for sid in self.h5['sample_ids'][:]]
        self.id_to_idx = {sid: i for i, sid in enumerate(all_ids)}

        # Validate that every requested sample_id actually exists
        missing = set(sample_ids_list) - set(self.id_to_idx)
        if missing:
            raise ValueError(f"[ExpressionDataset] The following sample_ids were not found in {h5_path}:\n"
                             + "\n".join(sorted(missing)))

        self.sample_ids = sample_ids_list
        self.scaler = joblib.load(scaler_path) if scaler_path else None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        # At this point we know sid is valid
        row_idx = self.id_to_idx[sid]
        data_row = self.h5['data'][row_idx]
        if self.scaler:
            data_row = self.scaler.transform(data_row.reshape(1, -1))[0]
        return torch.from_numpy(data_row).float()

class MultiOmicsDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]