import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import Dataset


def tsv_to_h5(tsv_path: str, h5_path: str, chunk_mb: int = 4):
    """
    Read a TSV with header sample_id \t feat1 \t feat2 ...  
    and write to an HDF5 file with:
      /sample_ids (n_samples,)   bytes
      /data       (n_samples, n_features) float32, chunked by rows, gzip-compressed
    """
    # First pass: read header and count samples
    with open(tsv_path, 'r') as f:
        header = f.readline().rstrip().split('\t')
        n_features = len(header) - 1

        sample_ids = []
        for line in f:
            sample_ids.append(line.split('\t', 1)[0])
    n_samples = len(sample_ids)
    sample_ids_arr = np.array(sample_ids, dtype='S')   # bytes

    
    bytes_per_row = n_features * 4  # float32 = 4 bytes
    rows_per_chunk = max(1, (chunk_mb * 1024**2) // bytes_per_row)

    # Create HDF5
    with h5py.File(h5_path, 'w') as h5:
        h5.create_dataset('sample_ids',
                           data=sample_ids_arr,
                           dtype=h5py.string_dtype(encoding='utf-8'))
        dset = h5.create_dataset('data',
                                 shape=(n_samples, n_features),
                                 dtype='float32',
                                 chunks=(rows_per_chunk, n_features),
                                 compression='gzip')
        # Second pass: fill data
        with open(tsv_path, 'r') as f:
            f.readline()  # skip header
            for i, line in enumerate(f):
                parts = line.rstrip().split('\t')
                dset[i, :] = np.asarray(parts[1:], dtype='float32')




def fit_scaler(h5_path, sample_ids, save_path):
    h5 = h5py.File(h5_path, 'r')
    all_ids = h5['sample_ids'][:]
    id_to_idx = {sid.decode('utf-8'): i for i, sid in enumerate(all_ids)}
    X = np.stack([h5['data'][id_to_idx[sid]] for sid in sample_ids])  # shape: (n_samples, n_features)
    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, save_path)
    h5.close()

