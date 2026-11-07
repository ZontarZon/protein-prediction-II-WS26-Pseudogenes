"""Shared paths, data loading, CV splits, MLP builder and training loop."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
WEBVIEWER = REPO / 'WebViewer'
DATA_DIR = WEBVIEWER / 'ModelsAndTrainers' / 'Datasets' / 'mlp_balanced'
SPROT = WEBVIEWER / 'ModelsAndTrainers' / 'Datasets' / 'uniprot_full_fasta' / 'uniprot_sprot.fasta.gz'
DOCS = REPO / 'docs' / 'mlp_experiments'
RESULTS = DOCS / 'results.jsonl'
N_FOLDS = 5


def load_table():
    rows = [line.rstrip('\n').split('\t') for line in open(DATA_DIR / 'subset.tsv')]
    header, rows = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in rows]


def load_data():
    """Return accessions, X (n,1024 float32), y (n,), cluster group ids (n,)."""
    table = load_table()
    acc = [r['accession'] for r in table]
    y = np.array([int(r['label']) for r in table], dtype=np.int64)
    emb = np.load(DATA_DIR / 'embeddings.npz')
    X = np.stack([emb[a] for a in acc]).astype(np.float32)
    clusters = {}
    for line in open(DATA_DIR / 'clusters_cluster.tsv'):
        rep, member = line.split()
        clusters[member] = rep
    reps = {rep: i for i, rep in enumerate(sorted(set(clusters.values())))}
    groups = np.array([reps[clusters[a]] for a in acc], dtype=np.int64)
    return acc, X, y, groups


def make_splits(y, groups, split_seed):
    """Outer StratifiedGroupKFold(5) + inner group split (1/8 of outer train) for early stopping."""
    from sklearn.model_selection import StratifiedGroupKFold
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
    splits = []
    for train, test in outer.split(np.zeros(len(y)), y, groups):
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=split_seed + 1000)
        itr, iva = next(inner.split(np.zeros(len(train)), y[train], groups[train]))
        splits.append({'train': train, 'test': test, 'inner_train': train[itr], 'inner_val': train[iva]})
    return splits


# ---------------------------------------------------------------- model
class ResidualBlock(nn.Module):
    def __init__(self, width, norm, act, dropout):
        super().__init__()
        self.body = nn.Sequential(*block(width, width, norm, act, dropout))

    def forward(self, x):
        return x + self.body(x)


def norm_layer(kind, width):
    return {'none': nn.Identity, 'layernorm': nn.LayerNorm, 'batchnorm': nn.BatchNorm1d}[kind](width)


def act_layer(kind):
    return {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[kind]()


def block(fan_in, width, norm, act, dropout):
    return [nn.Linear(fan_in, width), norm_layer(norm, width), act_layer(act), nn.Dropout(dropout)]


def build_mlp(hidden, dropout=0.1, norm='none', act='relu', residual=False, input_dropout=0.0, in_dim=1024, out_dim=2):
    layers = [nn.Dropout(input_dropout)] if input_dropout else []
    fan_in = in_dim
    for i, width in enumerate(hidden):
        if residual and i > 0 and width == fan_in:
            layers.append(ResidualBlock(width, norm, act, dropout))
        else:
            layers += block(fan_in, width, norm, act, dropout)
        fan_in = width
    layers.append(nn.Linear(fan_in, out_dim))
    return nn.Sequential(*layers)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- metrics
def metrics(y_true, prob1):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, roc_auc_score, average_precision_score)
    pred = (prob1 >= 0.5).astype(int)
    return {'auroc': float(roc_auc_score(y_true, prob1)), 'auprc': float(average_precision_score(y_true, prob1)),
            'acc': float(accuracy_score(y_true, pred)), 'bacc': float(balanced_accuracy_score(y_true, pred)),
            'mcc': float(matthews_corrcoef(y_true, pred)), 'f1': float(f1_score(y_true, pred))}


def fast_auroc(y, s):
    """Rank-based AUROC (ties averaged) on torch tensors."""
    from scipy.stats import rankdata
    y = y.cpu().numpy(); s = s.cpu().numpy()
    r = rankdata(s)
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---------------------------------------------------------------- training
DEFAULTS = dict(hidden=[256], dropout=0.1, norm='none', act='relu', residual=False, input_dropout=0.0,
                lr=1e-3, weight_decay=0.0, batch_size=64, epochs=100, patience=15, early_stopping=True,
                label_smoothing=0.0, standardize=False, optimizer='adamw', schedule='constant',
                swa=False, ensemble=1, mixup=0.0, refit=False)


def _predict(net, X):
    net.eval()
    with torch.no_grad():
        return net(X).softmax(-1)[:, 1]


def train_one(cfg, X, y, tr, va, te, seed, device):
    """Train on tr, early-stop on va (if enabled), return P(class 1) on te and info."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = X[tr], y[tr]
    if cfg['standardize']:
        mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    else:
        mu = torch.zeros(1, X.shape[1], device=device); sd = torch.ones(1, X.shape[1], device=device)
    norm = lambda a: (a - mu) / sd
    Xtr = norm(Xtr)
    Xva = norm(X[va]) if va is not None else None
    Xte = norm(X[te])
    net = build_mlp(cfg['hidden'], cfg['dropout'], cfg['norm'], cfg['act'], cfg['residual'], cfg['input_dropout'],
                    in_dim=X.shape[1]).to(device)
    opt_cls = torch.optim.AdamW if cfg['optimizer'] == 'adamw' else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    epochs, bs = int(cfg['epochs']), int(cfg['batch_size'])
    steps_per_epoch = math.ceil(len(tr) / bs)
    sched = None
    if cfg['schedule'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    swa_model = torch.optim.swa_utils.AveragedModel(net) if cfg['swa'] else None
    swa_start = int(epochs * 0.5)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    n = len(tr)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2 and cfg['norm'] == 'batchnorm':
                continue
            xb, yb = Xtr[idx], ytr[idx]
            if cfg['mixup']:
                lam = float(np.random.beta(cfg['mixup'], cfg['mixup']))
                j = torch.randperm(len(idx), device=device)
                out = net(lam * xb + (1 - lam) * xb[j])
                loss = lam * loss_fn(out, yb) + (1 - lam) * loss_fn(out, yb[j])
            else:
                loss = loss_fn(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(net)
        if cfg['early_stopping'] and va is not None:
            auc = fast_auroc(y[va], _predict(net, Xva))
            if auc > best + 1e-4:
                best, best_epoch, bad = auc, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg['patience']:
                    break
    if swa_model is not None:
        if cfg['norm'] == 'batchnorm':
            torch.optim.swa_utils.update_bn([(Xtr[i:i + bs],) for i in range(0, n, bs)], swa_model)
        final = swa_model
        best_epoch = epoch + 1
    elif best_state is not None:
        net.load_state_dict(best_state)
        final = net
    else:
        final = net
        best_epoch = epoch + 1
    return _predict(final, Xte).cpu().numpy(), {'best_epoch': best_epoch, 'epochs_run': epoch + 1,
                                                'val_auroc': best if best >= 0 else None, 'n_params': n_params(net)}


def run_cv(cfg, X, y, splits, seed, device):
    """Cross-validate cfg. X, y are torch tensors on device. Returns per-fold metrics + info."""
    cfg = {**DEFAULTS, **cfg}
    folds, infos = [], []
    yn = y.cpu().numpy()
    oof = np.full(len(yn), np.nan, dtype=np.float32)
    t0 = time.time()
    for k, sp in enumerate(splits):
        if cfg['early_stopping'] or cfg['swa']:
            tr, va = sp['inner_train'], sp['inner_val']
        else:
            tr, va = sp['train'], None
        tr_t = torch.as_tensor(tr, device=device); te_t = torch.as_tensor(sp['test'], device=device)
        va_t = torch.as_tensor(va, device=device) if va is not None else None
        probs = []
        for e in range(int(cfg['ensemble'])):
            p, info = train_one(cfg, X, y, tr_t, va_t, te_t, seed * 1000 + k * 10 + e, device)
            if cfg['refit'] and va is not None:
                # retrain on the full outer-train fold for the early-stopped epoch count
                full = {**cfg, 'epochs': info['best_epoch'], 'early_stopping': False, 'swa': False}
                p, _ = train_one(full, X, y, torch.as_tensor(sp['train'], device=device), None, te_t,
                                 seed * 1000 + k * 10 + e, device)
            probs.append(p); infos.append(info)
        oof[sp['test']] = np.mean(probs, 0)
        folds.append(metrics(yn[sp['test']], oof[sp['test']]))
    elapsed = time.time() - t0
    keys = folds[0].keys()
    summary = {f'{m}_mean': float(np.mean([f[m] for f in folds])) for m in keys}
    summary.update({f'{m}_std': float(np.std([f[m] for f in folds], ddof=1)) for m in keys})
    return {'config': cfg, 'n_params': infos[0]['n_params'] * int(cfg['ensemble']), 'folds': folds,
            'best_epochs': [i['best_epoch'] for i in infos], 'train_seconds': elapsed, '_oof': oof, **summary}


PRED_DIR = DATA_DIR / 'predictions'


def save_oof(record):
    """Store out-of-fold P(PE5) (one value per subset.tsv row) next to the data; strip it from the record."""
    oof = record.pop('_oof', None)
    if oof is not None and record.get('name'):
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PRED_DIR / f"{record['name']}_s{record['seed']}.npy", oof)
    return record


def append_result(record, path=RESULTS):
    record = save_oof(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as fh:
        fh.write(json.dumps(record) + '\n')


def read_results(path=RESULTS):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]
