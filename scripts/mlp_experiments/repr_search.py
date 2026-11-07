"""Phase 2: representation experiments on frozen per-residue ProtT5 embeddings.

Variants (all on the same 3 seeds x 5 grouped folds as phase 1, early stopping on the inner split):
  pooled   fixed pooled features -> phase-1 MLP family (mean control, mean/max/std, +extras, segments)
  la       light-attention pooling (Staerk et al. 2021), reduced width
  tf       tiny transformer encoder over residues, mean+max pooling
Usage:
  python repr_search.py features                 # build pooled feature cache from residues.h5
  python repr_search.py pooled  [--workers 6]
  python repr_search.py la      [--workers 2]
  python repr_search.py tf      [--workers 2]
  python repr_search.py custom --configs FILE    # list of {name, kind, cfg, seeds}
"""
import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np

from common import DATA_DIR, append_result, load_data, load_table, make_splits, metrics, read_results

FEATS = DATA_DIR / 'pooled_features.npz'
H5 = DATA_DIR / 'residues.h5'
SEEDS = (0, 1, 2)


# ------------------------------------------------------------------ features
def segment_means(R, n):
    L = len(R)
    edges = np.linspace(0, L, n + 1)
    out = []
    for j in range(n):
        lo, hi = int(math.floor(edges[j])), int(math.floor(edges[j + 1]))
        if hi <= lo:  # sequence shorter than n residues: use the residue at that relative position
            lo, hi = min(int(edges[j]), L - 1), min(int(edges[j]), L - 1) + 1
        out.append(R[lo:hi].mean(0))
    return np.concatenate(out)


def build_features():
    import h5py
    acc = [r['accession'] for r in load_table()]
    cols = {k: [] for k in ('mean', 'max', 'std', 'loglen', 'first16', 'last16', 'seg4', 'seg8')}
    with h5py.File(H5, 'r') as h5:
        for a in acc:
            R = h5[a][()].astype(np.float32)
            cols['mean'].append(R.mean(0)); cols['max'].append(R.max(0)); cols['std'].append(R.std(0))
            cols['loglen'].append(np.array([np.log(len(R))], np.float32))
            cols['first16'].append(R[:16].mean(0)); cols['last16'].append(R[-16:].mean(0))
            cols['seg4'].append(segment_means(R, 4)); cols['seg8'].append(segment_means(R, 8))
    np.savez(FEATS, **{k: np.stack(v).astype(np.float32) for k, v in cols.items()})
    ref = np.load(DATA_DIR / 'embeddings.npz')
    err = max(float(np.abs(np.stack(cols['mean'])[i] - ref[a]).max()) for i, a in enumerate(acc))
    print(f'wrote {FEATS}; max |mean(residues.h5) - embeddings.npz| = {err:.2e}')


FEATURE_SETS = {
    'mean': ['mean'],
    'mean_max_std': ['mean', 'max', 'std'],
    'mms_extras': ['mean', 'max', 'std', 'loglen', 'first16', 'last16'],
    'seg4': ['seg4'],
    'seg8': ['seg8'],
    'mean_seg4': ['mean', 'seg4'],
}


def feature_matrix(name):
    f = np.load(FEATS)
    return np.concatenate([f[k] for k in FEATURE_SETS[name]], 1)


# ------------------------------------------------------------------ residue models
def residue_store(device):
    """All residues concatenated on the GPU (fp16) + per-protein offsets/lengths in subset.tsv order."""
    import h5py
    import torch
    acc = [r['accession'] for r in load_table()]
    with h5py.File(H5, 'r') as h5:
        lens = np.array([h5[a].shape[0] for a in acc])
        flat = np.empty((lens.sum(), 1024), np.float16)
        off = np.concatenate([[0], np.cumsum(lens)[:-1]])
        for a, o, n in zip(acc, off, lens):
            flat[o:o + n] = h5[a][()]
    return torch.as_tensor(flat, device=device), torch.as_tensor(off, device=device), torch.as_tensor(lens, device=device)


def gather(store, idx, cap):
    """Padded batch [B, L, 1024] (float32) and mask [B, L] for protein indices idx (first `cap` residues)."""
    import torch
    R, off, lens = store
    L = torch.clamp(lens[idx], max=cap)
    Lmax = int(L.max())
    pos = torch.arange(Lmax, device=R.device)
    mask = pos[None, :] < L[:, None]
    rows = off[idx][:, None] + torch.minimum(pos[None, :], (L - 1)[:, None])
    return R[rows].float(), mask


def make_la(cfg):
    import torch
    from torch import nn

    class LightAttention(nn.Module):
        """Staerk et al. 2021, with a narrower conv (d channels) to stay at ~1-2M parameters."""
        def __init__(self, d, dropout, hidden, k=9):
            super().__init__()
            self.values = nn.Conv1d(1024, d, k, padding=k // 2)
            self.attention = nn.Conv1d(1024, d, k, padding=k // 2)
            self.dropout = nn.Dropout(dropout)
            self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, 2))

        def forward(self, x, mask):
            x = x.transpose(1, 2)                      # B, 1024, L
            m = mask[:, None, :]
            v = self.dropout(self.values(x))
            a = self.attention(x).float().masked_fill(~m, float('-inf'))
            pooled = (v.float() * a.softmax(-1)).sum(-1)
            vmax = v.masked_fill(~m, torch.finfo(v.dtype).min).max(-1).values
            return self.mlp(torch.cat([pooled, vmax.float()], -1))
    return LightAttention(cfg['d'], cfg['dropout'], cfg['hidden'])


def make_tf(cfg):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, d, heads, ff, dropout):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
            self.drop, self.heads, self.p = nn.Dropout(dropout), heads, dropout

        def forward(self, x, mask):
            B, L, d = x.shape
            q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :],
                                                 dropout_p=self.p if self.training else 0.0)
            x = x + self.drop(self.o(att.transpose(1, 2).reshape(B, L, d)))
            return x + self.drop(self.ff(self.n2(x)))

    class TinyTransformer(nn.Module):
        """Projection 1024->d, `layers` pre-LN blocks (no positional encoding; ProtT5 states are contextual), mean+max pool."""
        def __init__(self, d, layers, heads, dropout):
            super().__init__()
            self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(1024, d))
            self.blocks = nn.ModuleList(Block(d, heads, 2 * d, dropout) for _ in range(layers))
            self.norm = nn.LayerNorm(d)
            self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 2))

        def forward(self, x, mask):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h, mask)
            h = self.norm(h)
            m = mask[..., None]
            mean = (h * m).sum(1) / m.sum(1)
            mx = h.masked_fill(~m, -1e4).max(1).values
            return self.out(torch.cat([mean, mx], -1))
    return TinyTransformer(cfg['d'], cfg['layers'], cfg['heads'], cfg['dropout'])


def train_residue_model(cfg, store, y, tr, va, te, seed, device):
    import torch
    from torch import nn
    from common import fast_auroc
    torch.manual_seed(seed); np.random.seed(seed)
    net = (make_la if cfg['model'] == 'light_attention' else make_tf)(cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    bs, cap = cfg['batch_size'], cfg['cap']

    def predict(idx):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            srt = torch.argsort(store[2][idx])                 # length-sorted batches: little padding
            order = idx[srt]
            for i in range(0, len(order), 64):
                x, m = gather(store, order[i:i + 64], cap)
                out.append(net(x, m).float().softmax(-1)[:, 1])
        p = torch.empty(len(idx), device=device)
        p[srt] = torch.cat(out)
        return p

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(cfg['epochs']):
        net.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x, m = gather(store, idx, cap)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss_fn(net(x, m).float(), y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        auc = fast_auroc(y[va], predict(va))
        if auc > best + 1e-4:
            best, best_epoch, bad = auc, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg['patience']:
                break
    net.load_state_dict(best_state)
    return predict(te).cpu().numpy(), best_epoch, sum(p.numel() for p in net.parameters())


def residue_cv(cfg, store, y, splits, seed, device):
    import torch
    yn = y.cpu().numpy()
    oof = np.full(len(yn), np.nan, np.float32)
    folds, epochs, t0 = [], [], time.time()
    for k, sp in enumerate(splits):
        t = lambda a: torch.as_tensor(a, device=device)
        p, ep, n = train_residue_model(cfg, store, y, t(sp['inner_train']), t(sp['inner_val']), t(sp['test']),
                                       seed * 1000 + k * 10, device)
        oof[sp['test']] = p
        folds.append(metrics(yn[sp['test']], p)); epochs.append(ep)
    out = {'config': cfg, 'n_params': n, 'folds': folds, 'best_epochs': epochs, 'train_seconds': time.time() - t0, '_oof': oof}
    for m in folds[0]:
        out[f'{m}_mean'] = float(np.mean([f[m] for f in folds]))
        out[f'{m}_std'] = float(np.std([f[m] for f in folds], ddof=1))
    return out


# ------------------------------------------------------------------ workers
_S = {}


def _init(need_residues):
    import torch
    torch.set_num_threads(2)
    _, X, y, groups = load_data()
    _S.update(y=torch.as_tensor(y, device='cuda'), yn=y, groups=groups, splits={}, X={})
    if need_residues:
        _S['store'] = residue_store('cuda')


def _work(job):
    import torch
    from common import run_cv
    s = job['seed']
    if s not in _S['splits']:
        _S['splits'][s] = make_splits(_S['yn'], _S['groups'], s)
    cfg = job['cfg']
    if cfg['model'] == 'pooled_mlp':
        f = cfg['features']
        if f not in _S['X']:
            _S['X'][f] = torch.as_tensor(feature_matrix(f), device='cuda')
        mlp = {k: v for k, v in cfg.items() if k not in ('model', 'features')}
        res = run_cv(mlp, _S['X'][f], _S['y'], _S['splits'][s], s, 'cuda')
        res['config'] = {'model': 'pooled_mlp', 'features': f, **res['config']}
    else:
        res = residue_cv(cfg, _S['store'], _S['y'], _S['splits'][s], s, 'cuda')
    torch.cuda.empty_cache()
    return {'phase': 'repr', 'name': job['name'], 'split_seed': s, 'seed': s, **res}


def run(jobs, workers, need_residues):
    done = {(r.get('name'), r['seed']) for r in read_results()}
    jobs = [j for j in jobs if (j['name'], j['seed']) not in done]
    print(f'{len(jobs)} jobs', flush=True)
    if not jobs:
        return
    t0 = time.time()
    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(need_residues,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, jobs), 1):
            append_result(res)
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {res['name']} s{res['seed']} auroc={res['auroc_mean']:.4f}±{res['auroc_std']:.4f} "
                  f"mcc={res['mcc_mean']:.3f} params={res['n_params']} epochs={res['best_epochs']} t={res['train_seconds']:.0f}s", flush=True)


MLP_BASE = dict(hidden=[512, 512], norm='layernorm', act='gelu', weight_decay=0.0, batch_size=32, epochs=100,
                patience=15, early_stopping=True, optimizer='adamw', refit=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['features', 'pooled', 'la', 'tf', 'custom'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--configs')
    ap.add_argument('--seeds', default='0,1,2', help='comma list; e.g. 0 to screen, then rerun winners on 1,2')
    ap.add_argument('--only', default=None, help='comma list of run names to keep')
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(','))
    if args.phase == 'features':
        return build_features()
    jobs = []
    if args.phase == 'pooled':
        for f in FEATURE_SETS:
            for dr in (0.3, 0.5):
                for lr in (1e-4, 3e-4):
                    for st in (False, True):
                        name = f'pool_{f}_d{dr}_lr{lr:g}_z{int(st)}'
                        cfg = {'model': 'pooled_mlp', 'features': f, **MLP_BASE, 'dropout': dr, 'lr': lr, 'standardize': st}
                        jobs += [dict(name=name, cfg=cfg, seed=s) for s in SEEDS]
    elif args.phase == 'la':
        for d in (64, 128):
            for dr in (0.25, 0.5):
                for lr in (1e-4, 3e-4):
                    cfg = dict(model='light_attention', d=d, hidden=256, dropout=dr, lr=lr, weight_decay=1e-2,
                               batch_size=32, epochs=30, patience=8, cap=4096)
                    jobs += [dict(name=f'la_d{d}_dr{dr}_lr{lr:g}', cfg=cfg, seed=s) for s in SEEDS]
    elif args.phase == 'tf':
        for d, layers in ((128, 2), (256, 2)):
            for dr in (0.1, 0.3):
                for lr in (1e-4, 3e-4):
                    cfg = dict(model='transformer', d=d, layers=layers, heads=4, dropout=dr, lr=lr, weight_decay=1e-2,
                               batch_size=32, epochs=30, patience=8, cap=2048)
                    jobs += [dict(name=f'tf_d{d}_L{layers}_dr{dr}_lr{lr:g}', cfg=cfg, seed=s) for s in SEEDS]
    else:
        for item in json.load(open(args.configs)):
            jobs += [dict(name=item['name'], cfg=item['cfg'], seed=s) for s in item.get('seeds', SEEDS)]
    jobs = [j for j in jobs if j['seed'] in seeds and (args.only is None or j['name'] in args.only.split(','))]
    need = any(j['cfg']['model'] != 'pooled_mlp' for j in jobs)
    run(jobs, args.workers, need)


if __name__ == '__main__':
    main()
