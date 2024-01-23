import numpy as np
import json

PACK_SIZE = 1000

for start in range(0, 100000, PACK_SIZE):
    print(start)
    nets, labels = [], []
    for i in range(start, start + PACK_SIZE):
        try:
            nets.append(np.load(f'/rds/project/rds-eWkDxBhxBrQ/neel/ctc_new/{i}/epoch_20.npy', allow_pickle=True).item())
            with open(f'/rds/project/rds-eWkDxBhxBrQ/neel/ctc/{i}/run_data.json', 'rb') as f:
                run_file = json.load(f)
                labels.append(run_file['hyperparameters'])
        except Exception:
            print("Unknown error at net", i)
            print(Exception)
            
    np.savez(f'/rds/project/rds-eWkDxBhxBrQ/neel/ctc_new/packed_nets_{start}_{start+PACK_SIZE}.npz',
        nets=np.array(nets),
        labels=np.array(labels)
    )