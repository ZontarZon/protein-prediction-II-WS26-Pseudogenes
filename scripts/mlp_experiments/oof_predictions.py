"""Recompute out-of-fold predictions for phase-1 reference configs (phase 1 stored metrics only).

Writes Datasets/mlp_balanced/predictions/<name>_s<seed>.npy and checks that the recomputed fold
metrics match results.jsonl (GPU nondeterminism can move the last digits). Does not append results.
"""
import numpy as np
import torch

from common import load_data, make_splits, read_results, run_cv, PRED_DIR

NAMES = ['a_repo_defaults', 'b_repo_head_sensible', 'T1_fixed6', 'T1_fixed6_ens10']


def main():
    _, X, y, g = load_data()
    Xt, yt = torch.as_tensor(X, device='cuda'), torch.as_tensor(y, device='cuda')
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    rows = {(r['name'], r['seed']): r for r in reversed(read_results()) if r.get('name') in NAMES}
    for name in NAMES:
        for s in (0, 1, 2):
            ref = rows[(name, s)]
            res = run_cv(ref['config'], Xt, yt, make_splits(y, g, s), s, 'cuda')
            np.save(PRED_DIR / f'{name}_s{s}.npy', res['_oof'])
            print(f"{name} s{s}: stored AUROC {ref['auroc_mean']:.4f}, recomputed {res['auroc_mean']:.4f}", flush=True)


if __name__ == '__main__':
    main()
