"""Cross-validated baselines and MLP search on cached ProtT5 embeddings.

Every run appends one JSON line to docs/mlp_experiments/results.jsonl.
Phases:
  baselines  (a) repo head + trainer defaults, (b) repo head + sensible training, (c) logistic regression
  random     broad random search (--n configs)
  refine     local perturbations around the current top configs (--n configs)
  widen      residual MLPs, seed ensembles, SWA, cosine schedule, mixup, longer training
  seeds      re-run baselines and the top --top configs on split/model seeds 1 and 2
"""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import random
import time

import numpy as np

from common import DEFAULTS, append_result, load_data, make_splits, read_results

BASELINE_A = dict(hidden=[256], dropout=0.1, norm='none', act='relu', lr=1e-3, weight_decay=0.0, batch_size=2,
                  epochs=3, early_stopping=False, optimizer='adam', standardize=False)
BASELINE_B = dict(hidden=[256], dropout=0.1, norm='none', act='relu', lr=1e-3, weight_decay=0.0, batch_size=64,
                  epochs=100, patience=15, early_stopping=True, optimizer='adam', standardize=False)
SPACE = dict(depth=[1, 2, 3, 4], width=[64, 128, 256, 512, 1024], dropout=[0.0, 0.1, 0.3, 0.5],
             norm=['none', 'layernorm', 'batchnorm'], act=['relu', 'gelu'], lr=[1e-4, 3e-4, 1e-3],
             weight_decay=[0.0, 1e-4, 1e-2], batch_size=[32, 64, 128], label_smoothing=[0.0, 0.1],
             standardize=[False, True])


def random_config(rng):
    c = {k: rng.choice(v) for k, v in SPACE.items()}
    depth, width = c.pop('depth'), c.pop('width')
    c['hidden'] = [width] * depth
    c.update(epochs=100, patience=15, early_stopping=True, optimizer='adamw')
    return c


def key(cfg):
    cfg = {**DEFAULTS, **cfg}
    return json.dumps(cfg, sort_keys=True)


# ------------------------------------------------------------------ workers
_STATE = {}


def _init(device):
    import torch
    torch.set_num_threads(2)
    acc, X, y, groups = load_data()
    _STATE.update(X=torch.as_tensor(X, device=device), y=torch.as_tensor(y, device=device), Xn=X, yn=y,
                  groups=groups, device=device, splits={})


def _splits(split_seed):
    if split_seed not in _STATE['splits']:
        _STATE['splits'][split_seed] = make_splits(_STATE['yn'], _STATE['groups'], split_seed)
    return _STATE['splits'][split_seed]


def logreg_cv(splits, X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from common import metrics
    folds, chosen = [], []
    t0 = time.time()
    for sp in splits:
        best = (-1, None)
        for C in [1e-3, 1e-2, 1e-1, 1.0]:
            sc = StandardScaler().fit(X[sp['inner_train']])
            m = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(sc.transform(X[sp['inner_train']]), y[sp['inner_train']])
            auc = metrics(y[sp['inner_val']], m.predict_proba(sc.transform(X[sp['inner_val']]))[:, 1])['auroc']
            best = max(best, (auc, C), key=lambda t: t[0])
        sc = StandardScaler().fit(X[sp['train']])
        m = LogisticRegression(C=best[1], max_iter=5000, random_state=seed).fit(sc.transform(X[sp['train']]), y[sp['train']])
        folds.append(metrics(y[sp['test']], m.predict_proba(sc.transform(X[sp['test']]))[:, 1]))
        chosen.append(best[1])
    out = {'config': {'model': 'logistic_regression', 'C_grid': [1e-3, 1e-2, 1e-1, 1.0], 'standardize': True},
           'chosen_C': chosen, 'n_params': X.shape[1] + 1, 'folds': folds, 'train_seconds': time.time() - t0}
    for m in folds[0]:
        out[f'{m}_mean'] = float(np.mean([f[m] for f in folds]))
        out[f'{m}_std'] = float(np.std([f[m] for f in folds], ddof=1))
    return out


def svm_cv(splits, X, y, seed):
    """RBF-SVM reference (not an MLP): is the plateau a property of the features?"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from common import metrics
    folds, chosen = [], []
    t0 = time.time()
    grid = [(C, g) for C in (0.3, 1.0, 3.0, 10.0) for g in ('scale', 0.3 / X.shape[1], 3.0 / X.shape[1])]
    for sp in splits:
        sc = StandardScaler().fit(X[sp['inner_train']])
        best = (-1, None)
        for C, g in grid:
            m = SVC(C=C, gamma=g, random_state=seed).fit(sc.transform(X[sp['inner_train']]), y[sp['inner_train']])
            auc = metrics(y[sp['inner_val']], m.decision_function(sc.transform(X[sp['inner_val']])))['auroc']
            best = max(best, (auc, (C, g)), key=lambda t: t[0])
        sc = StandardScaler().fit(X[sp['train']])
        m = SVC(C=best[1][0], gamma=best[1][1], probability=True, random_state=seed).fit(sc.transform(X[sp['train']]), y[sp['train']])
        folds.append(metrics(y[sp['test']], m.predict_proba(sc.transform(X[sp['test']]))[:, 1]))
        chosen.append(best[1])
    out = {'config': {'model': 'svm_rbf', 'grid': [list(map(str, g)) for g in grid], 'standardize': True},
           'chosen': [list(map(str, c)) for c in chosen], 'n_params': None, 'folds': folds, 'train_seconds': time.time() - t0}
    for m in folds[0]:
        out[f'{m}_mean'] = float(np.mean([f[m] for f in folds]))
        out[f'{m}_std'] = float(np.std([f[m] for f in folds], ddof=1))
    return out


def _work(job):
    from common import run_cv
    splits = _splits(job['split_seed'])
    if job['cfg'] == 'logreg':
        res = logreg_cv(splits, _STATE['Xn'], _STATE['yn'], job['seed'])
    elif job['cfg'] == 'svm':
        res = svm_cv(splits, _STATE['Xn'], _STATE['yn'], job['seed'])
    else:
        res = run_cv(job['cfg'], _STATE['X'], _STATE['y'], splits, job['seed'], _STATE['device'])
    return {'phase': job['phase'], 'name': job.get('name'), 'split_seed': job['split_seed'], 'seed': job['seed'], **res}


def run_jobs(jobs, workers):
    if not jobs:
        return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(workers, initializer=_init, initargs=('cuda',)) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, jobs), 1):
            append_result(res)
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {res['phase']} {res.get('name') or ''} "
                  f"auroc={res['auroc_mean']:.4f}±{res['auroc_std']:.4f} mcc={res['mcc_mean']:.3f} "
                  f"params={res['n_params']} t={res['train_seconds']:.0f}s", flush=True)


# ------------------------------------------------------------------ phases
def done_keys(seed=0):
    return {key(r['config']) for r in read_results() if r['seed'] == seed and isinstance(r['config'], dict) and 'hidden' in r['config']}


def top_configs(n, phases=('random', 'refine', 'widen'), seed=0):
    rows = [r for r in read_results() if r['phase'] in phases and r['seed'] == seed]
    rows.sort(key=lambda r: -r['auroc_mean'])
    out, seen = [], set()
    for r in rows:
        k = key(r['config'])
        if k not in seen:
            seen.add(k); out.append(r)
        if len(out) == n:
            break
    return out


def perturb(cfg, rng):
    c = copy.deepcopy(cfg)
    for _ in range(rng.choice([1, 2])):
        field = rng.choice(['depth', 'width', 'dropout', 'norm', 'act', 'lr', 'weight_decay', 'batch_size',
                            'label_smoothing', 'standardize', 'input_dropout'])
        if field == 'depth':
            d = max(1, min(4, len(c['hidden']) + rng.choice([-1, 1])))
            c['hidden'] = [c['hidden'][0]] * d
        elif field == 'width':
            c['hidden'] = [max(32, min(2048, int(c['hidden'][0] * rng.choice([0.5, 2]))))] * len(c['hidden'])
        elif field == 'dropout':
            c['dropout'] = rng.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        elif field == 'lr':
            c['lr'] = rng.choice([5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3])
        elif field == 'weight_decay':
            c['weight_decay'] = rng.choice([0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1])
        elif field == 'batch_size':
            c['batch_size'] = rng.choice([16, 32, 64, 128, 256])
        elif field == 'label_smoothing':
            c['label_smoothing'] = rng.choice([0.0, 0.05, 0.1, 0.2])
        elif field == 'input_dropout':
            c['input_dropout'] = rng.choice([0.0, 0.1, 0.2, 0.3])
        elif field == 'standardize':
            c['standardize'] = not c.get('standardize', False)
        else:
            c[field] = rng.choice(SPACE[field])
    c['patience'] = 20
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['baselines', 'random', 'refine', 'widen', 'seeds', 'custom'])
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rng', type=int, default=0)
    ap.add_argument('--configs', help='custom: JSON file with list of {name, cfg}')
    args = ap.parse_args()
    rng = random.Random(args.rng)
    jobs = []
    if args.phase == 'baselines':
        jobs = [dict(phase='baseline', name='a_repo_defaults', cfg=BASELINE_A, split_seed=0, seed=0),
                dict(phase='baseline', name='b_repo_head_sensible', cfg=BASELINE_B, split_seed=0, seed=0),
                dict(phase='baseline', name='c_logreg', cfg='logreg', split_seed=0, seed=0)]
        # (b+) the repo architecture with tuned optimisation only, so architecture gains are isolated.
        for lr in [1e-4, 3e-4, 1e-3]:
            for wd in [0.0, 1e-2]:
                for st in [False, True]:
                    cfg = {**BASELINE_B, 'lr': lr, 'weight_decay': wd, 'standardize': st, 'optimizer': 'adamw'}
                    jobs.append(dict(phase='baseline_tuned', name=f'b+_lr{lr}_wd{wd}_std{int(st)}', cfg=cfg, split_seed=0, seed=0))
    elif args.phase == 'random':
        seen = done_keys()
        while len(jobs) < args.n:
            c = random_config(rng)
            if key(c) not in seen:
                seen.add(key(c)); jobs.append(dict(phase='random', cfg=c, split_seed=0, seed=0))
    elif args.phase == 'refine':
        seen = done_keys()
        parents = [r['config'] for r in top_configs(args.top)]
        tries = 0
        while len(jobs) < args.n and tries < 100 * args.n:
            tries += 1
            c = perturb(rng.choice(parents), rng)
            if key(c) not in seen:
                seen.add(key(c)); jobs.append(dict(phase='refine', cfg=c, split_seed=0, seed=0))
    elif args.phase == 'widen':
        seen = done_keys()
        bases = [r['config'] for r in top_configs(args.top)]
        for i, b in enumerate(bases):
            variants = {
                'residual': {**b, 'residual': True, 'hidden': [b['hidden'][0]] * max(2, len(b['hidden']))},
                'ens5': {**b, 'ensemble': 5},
                'swa': {**b, 'swa': True, 'early_stopping': False, 'epochs': 60},
                'cosine': {**b, 'schedule': 'cosine', 'epochs': 100, 'patience': 30},
                'long_patience': {**b, 'epochs': 300, 'patience': 50},
                'mixup': {**b, 'mixup': 0.4},
                'input_dropout': {**b, 'input_dropout': 0.2},
                'refit': {**b, 'refit': True},
                'ens5_refit': {**b, 'ensemble': 5, 'refit': True},
            }
            for vname, c in variants.items():
                if key(c) not in seen:
                    seen.add(key(c)); jobs.append(dict(phase='widen', name=f'top{i + 1}_{vname}', cfg=c, split_seed=0, seed=0))
        # repo head with a seed ensemble, so the ensemble effect is not credited to architecture
        jobs.append(dict(phase='widen', name='b_ens5', cfg={**BASELINE_B, 'ensemble': 5}, split_seed=0, seed=0))
        jobs.append(dict(phase='widen', name='b_refit', cfg={**BASELINE_B, 'refit': True}, split_seed=0, seed=0))
    elif args.phase == 'seeds':
        tops = top_configs(args.top, phases=('random', 'refine', 'widen', 'custom'))
        base = [r for r in read_results() if r['phase'] in ('baseline', 'baseline_tuned') and r['seed'] == 0]
        ident = lambda r: r['name'] if r['config'].get('model') == 'logistic_regression' else key(r['config'])
        have = {(ident(r), r['seed']) for r in read_results()}
        for s in (1, 2):
            for r in base:
                cfg = 'logreg' if r['config'].get('model') == 'logistic_regression' else r['config']
                if (ident(r), s) not in have:
                    jobs.append(dict(phase=r['phase'], name=r['name'], cfg=cfg, split_seed=s, seed=s))
            for r in tops:
                if (key(r['config']), s) not in have:
                    name = r.get('name') or 'cfg_' + hashlib.md5(key(r['config']).encode()).hexdigest()[:8]
                    jobs.append(dict(phase='seeds', name=name, cfg=r['config'], split_seed=s, seed=s))
    elif args.phase == 'custom':
        for item in json.load(open(args.configs)):
            for s in item.get('seeds', [0]):
                jobs.append(dict(phase=item.get('phase', 'custom'), name=item['name'], cfg=item['cfg'], split_seed=s, seed=s))
    print(f'{args.phase}: {len(jobs)} jobs', flush=True)
    run_jobs(jobs, args.workers)


if __name__ == '__main__':
    main()
