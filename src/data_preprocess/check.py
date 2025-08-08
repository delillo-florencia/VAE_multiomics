import random
import numpy as np
import h5py

def verify_random_samples(tsv_path, h5_path, n_checks=5):
    # Read the TSV into a dict for quick lookup
    tsv_dict = {}
    with open(tsv_path) as f:
        header = f.readline().split('\t')
        for line in f:
            parts = line.rstrip().split('\t')
            tsv_dict[parts[0]] = np.array(parts[1:], dtype='float32')

    with h5py.File(h5_path, 'r') as h5:
        h5_ids = [sid.decode('utf-8') for sid in h5['sample_ids'][:]]
        for sid in random.sample(h5_ids, n_checks):
            i = h5_ids.index(sid)
            h5_row = h5['data'][i, :]
            tsv_row = tsv_dict[sid]
            assert np.allclose(h5_row, tsv_row), f"Mismatch for {sid}"
    print(f"All {n_checks} random checks passed.")

