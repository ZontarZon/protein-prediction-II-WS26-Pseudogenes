"""Regenerate docs/mlp_experiments/report.html and its PNG charts from results.jsonl."""
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _f in ['/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)

from common import DATA_DIR, DOCS, N_FOLDS, load_table, make_splits, read_results  # noqa: E402
from search import key  # noqa: E402

# muted palette (validated reference slots 1-3) + neutral ink
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
GREY, INK, INK2, SURF = '#9a9893', '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.edgecolor': '#cfcdc8', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ecebe7',
                     'grid.linewidth': 0.8, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'normal',
                     'axes.titlecolor': INK, 'legend.frameon': False,
                     'font.sans-serif': ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']})
METRICS = [('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('acc', 'Accuracy'), ('bacc', 'Balanced acc.'), ('mcc', 'MCC'), ('f1', 'F1')]
CHOSEN = DOCS / 'chosen.json'


def ident(r):
    c = r['config']
    return r['name'] if 'model' in c else key(c)


def group_runs(results):
    """config identity -> {seed: record} (first record per seed)."""
    g = defaultdict(dict)
    for r in results:
        g[ident(r)].setdefault(r['seed'], r)
    return g


def pooled(runs, seeds=(0, 1, 2)):
    """Per-fold metric arrays pooled over the given seeds (None if a seed is missing)."""
    if not all(s in runs for s in seeds):
        return None
    return {m: np.array([f[m] for s in seeds for f in runs[s]['folds']]) for m, _ in METRICS}


def fmt(mean, std, d=4):
    return f'{mean:.{d}f} ± {std:.{d}f}'


def describe(c):
    if c.get('model') == 'logistic_regression':
        return 'LogisticRegression（z-score，C 由内层验证选）'
    if c.get('model') == 'svm_rbf':
        return 'RBF-SVM（z-score，C/gamma 由内层验证选；非 MLP 参照）'
    parts = [f"隐藏层 {c['hidden']}", f"dropout {c['dropout']}", f"norm {c['norm']}", c['act'],
             f"lr {c['lr']:g}", f"wd {c['weight_decay']:g}", f"bs {c['batch_size']}", f"{c['optimizer']}"]
    if c.get('standardize'): parts.append('输入 z-score')
    if c.get('label_smoothing'): parts.append(f"label smoothing {c['label_smoothing']}")
    if c.get('input_dropout'): parts.append(f"input dropout {c['input_dropout']}")
    if c.get('residual'): parts.append('残差')
    if c.get('ensemble', 1) > 1: parts.append(f"{c['ensemble']} 种子集成")
    if c.get('swa'): parts.append('SWA')
    if c.get('schedule') == 'cosine': parts.append('cosine')
    if c.get('mixup'): parts.append(f"mixup {c['mixup']}")
    if c.get('refit'): parts.append('早停后全量重训')
    parts.append(f"最多 {c['epochs']} epoch" + (f"，patience {c['patience']}" if c.get('early_stopping') else '，无早停'))
    return '，'.join(parts)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=150)
    plt.close(fig)
    return name


# ------------------------------------------------------------------ charts
def chart_lengths(table):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bins = np.logspace(np.log10(10), np.log10(40000), 50)
    for label, name, col in [(0, 'PE 1–2（real，类 0）', C1), (1, 'PE 5（fake，类 1）', C2)]:
        L = [int(r['length']) for r in table if int(r['label']) == label]
        ax.hist(L, bins=bins, histtype='step', linewidth=2, color=col, label=f'{name}，中位数 {int(np.median(L))}')
    ax.set_xscale('log'); ax.set_xlabel('序列长度（残基，对数轴）'); ax.set_ylabel('条目数')
    ax.set_title('序列长度分布'); ax.legend(fontsize=9)
    return save(fig, 'lengths.png')


def chart_scatter(search_rows, ref_b, ref_a):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for phase, col, lab in [('random', GREY, '随机搜索'), ('refine', C1, '局部细化'), ('widen', C3, '扩展（残差/集成/SWA…）')]:
        rows = [r for r in search_rows if r['phase'] == phase]
        if rows:
            ax.scatter([r['n_params'] for r in rows], [r['auroc_mean'] for r in rows], s=16, color=col,
                       alpha=0.75 if phase != 'random' else 0.5, edgecolor=SURF, linewidth=0.5, label=lab, zorder=3)
    ax.axhline(ref_b, color=C2, linewidth=1.5, label=f'基线 (b) {ref_b:.4f}', zorder=2)
    ax.axhline(ref_a, color=C2, linewidth=1.2, linestyle='--', label=f'基线 (a) {ref_a:.4f}', zorder=2)
    ax.set_xscale('log'); ax.set_xlabel('参数量（对数轴）'); ax.set_ylabel('5 折平均 AUROC（seed 0）')
    lo = np.percentile([r['auroc_mean'] for r in search_rows], 2)
    ax.set_ylim(lo - 0.002, max(r['auroc_mean'] for r in search_rows) + 0.002)
    ax.set_title('AUROC 与参数量'); ax.legend(fontsize=8, loc='lower left', ncol=2)
    return save(fig, 'auroc_vs_params.png')


def box(ax, rows, field, order, title, xl):
    data = [[r['auroc_mean'] for r in rows if field(r['config']) == v] for v in order]
    keep = [(str(v), d) for v, d in zip(order, data) if d]
    bp = ax.boxplot([d for _, d in keep], tick_labels=[f'{v}\n(n={len(d)})' for v, d in keep], widths=0.55,
                    patch_artist=True, showfliers=False, medianprops=dict(color=INK, linewidth=1.5),
                    whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    for p in bp['boxes']:
        p.set(facecolor='#d6e4f5', edgecolor=C1, linewidth=1)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xl); ax.tick_params(axis='x', labelsize=8)


def chart_depth_width(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    box(axes[0], rows, lambda c: len(c['hidden']), [1, 2, 3, 4], '按隐藏层数', '隐藏层数')
    box(axes[1], rows, lambda c: c['hidden'][0], [64, 128, 256, 512, 1024], '按宽度', '每层宽度')
    axes[0].set_ylabel('5 折平均 AUROC')
    return save(fig, 'auroc_by_depth_width.png')


def chart_effects(rows):
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.2), sharey=True)
    box(axes[0], rows, lambda c: c['dropout'], [0.0, 0.1, 0.3, 0.5], 'Dropout', 'p')
    box(axes[1], rows, lambda c: c['norm'], ['none', 'layernorm', 'batchnorm'], '归一化层', '')
    box(axes[2], rows, lambda c: bool(c['standardize']), [False, True], '输入 z-score', '')
    box(axes[3], rows, lambda c: c['lr'], [1e-4, 3e-4, 1e-3], '学习率', 'lr')
    axes[0].set_ylabel('5 折平均 AUROC')
    return save(fig, 'auroc_effects.png')


def chart_paired(rows):
    """rows: list of (label, per-fold AUROC differences)."""
    fig, ax = plt.subplots(figsize=(6.8, 0.9 + 0.75 * len(rows)))
    for i, (lab, d) in enumerate(rows):
        col = [C1, C3, GREY][i % 3]
        y = np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d))
        ax.scatter(d * 100, y, s=22, color=col, edgecolor=SURF, linewidth=0.5, zorder=3)
        ax.plot([d.mean() * 100] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2, zorder=4)
        ax.annotate(f'均值 {d.mean() * 100:+.2f}', (d.mean() * 100, i - 0.3), xytext=(4, 0), textcoords='offset points', ha='left', va='center', fontsize=8, color=INK2)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.invert_yaxis()
    ax.set_xlabel('每折 AUROC 差（百分点；3 个种子 × 5 折 = 15 个配对）'); ax.set_title('逐折配对差')
    return save(fig, 'paired_diff.png')


# ------------------------------------------------------------------ phase 2
FAMILIES = [('pool_mean_', '均值池化（对照，同协议）'), ('pool_mean_max_std_', '均值+最大+标准差（3072 维）'),
            ('pool_mms_extras_', '均值+最大+标准差+log 长度+首/尾 16 残基均值（5121 维）'),
            ('pool_seg4_', '4 段均值（4096 维）'), ('pool_seg8_', '8 段均值（8192 维）'), ('pool_mean_seg4_', '均值+4 段均值（5120 维）'),
            ('la_', '轻注意力池化（Stärk 2021，窄卷积）'), ('tf_', '小型 Transformer（1–2 层）')]


def family_of(name):
    # longest matching prefix wins (pool_mean_ is a prefix of pool_mean_max_std_)
    best = None
    for prefix, label in FAMILIES:
        if name.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            if prefix == 'pool_mean_' and not name[len(prefix):].startswith('d'):
                continue
            best = (prefix, label)
    return best


def describe_repr(c):
    if c['model'] == 'pooled_mlp':
        return f"MLP {c['hidden']}，LN，GELU，dropout {c['dropout']}，lr {c['lr']:g}" + ('，z-score' if c.get('standardize') else '') + '，早停后全量重训'
    if c['model'] == 'light_attention':
        return f"conv k9 → {c['d']} 通道（值/注意力），softmax 加权和 ⊕ max → MLP 256×2，dropout {c['dropout']}，lr {c['lr']:g}，≤{c['epochs']} epoch，截断 {c['cap']}"
    return f"1024→{c['d']}，{c['layers']} 层 × {c['heads']} 头，均值 ⊕ max 池化，dropout {c['dropout']}，lr {c['lr']:g}，≤{c['epochs']} epoch，截断 {c['cap']}"


def load_oof(name, seed):
    from common import PRED_DIR
    f = PRED_DIR / f'{name}_s{seed}.npy'
    return np.load(f) if f.exists() else None


def repr_section(results, groups, b_p, dep_p, charts):
    rows = [r for r in results if r['phase'] == 'repr']
    by = defaultdict(dict)
    for r in rows:
        by[r['name']].setdefault(r['seed'], r)
    picks = []
    for prefix, label in FAMILIES:
        cands = [n for n in by if family_of(n) and family_of(n)[0] == prefix and 0 in by[n]]
        if not cands:
            continue
        n = max(cands, key=lambda n: by[n][0]['auroc_mean'])           # select on seed 0 only
        picks.append((label, n, len(cands)))
    trs, paired = [], []
    for label, n, ncand in picks:
        runs = by[n]
        p = pooled(runs)
        r0 = runs[0]
        if p is None:
            trs.append([label, html.escape(describe_repr(r0['config'])), '—', f"{r0['auroc_mean']:.4f}（仅 seed 0）"] + ['—'] * 8)
            continue
        d, dd = p['auroc'] - b_p['auroc'], p['auroc'] - dep_p['auroc']
        dm = p['mcc'] - b_p['mcc']
        secs = np.mean([runs[s]['train_seconds'] for s in SEEDS3])
        trs.append([label, html.escape(describe_repr(r0['config'])) + f'<br><span class="note">{ncand} 个候选中按 seed 0 选出</span>',
                    f"{r0['n_params']:,}", ' / '.join(f"{runs[s]['auroc_mean']:.4f}" for s in SEEDS3),
                    fmt(p['auroc'].mean(), p['auroc'].std(ddof=1)), f"{p['auprc'].mean():.4f}", f"{p['mcc'].mean():.3f}", f"{p['bacc'].mean():.4f}",
                    f"{d.mean() * 100:+.2f}（{int((d > 0).sum())}/15）", f"{dm.mean() * 100:+.2f}", f"{dd.mean() * 100:+.2f}（{int((dd > 0).sum())}/15）", f'{secs:.0f}'])
        paired.append((label.split('（')[0], d))
    charts['repr'] = chart_paired_named(paired, 'repr_paired_diff.png', '表示层变体 − (b)：逐折 AUROC 差')
    hdr = ['变体', '设置', '参数量', 'AUROC s0/s1/s2', 'AUROC（15 折）', 'AUPRC', 'MCC', '平衡准确率', 'ΔAUROC vs (b)（胜出折）', 'ΔMCC vs (b)', 'ΔAUROC vs 部署头', '训练秒/种子']
    allrows = sorted(by.items(), key=lambda t: -(pooled(t[1])['auroc'].mean() if pooled(t[1]) else t[1][0]['auroc_mean']))
    grid = [[html.escape(n), ' / '.join(f"{runs[s]['auroc_mean']:.4f}" if s in runs else '—' for s in SEEDS3),
             f"{pooled(runs)['auroc'].mean():.4f}" if pooled(runs) else '—', f"{runs[min(runs)]['n_params']:,}"] for n, runs in allrows]
    target = [t for t in picks if pooled(by[t[1]]) is not None]
    best_name = max(target, key=lambda t: pooled(by[t[1]])['auroc'].mean())[1] if target else None
    html_out = f"""<h2>5. 表示层实验（逐残基嵌入）</h2>
<p>第 1 阶段的分类头搜索饱和，所以这里改动表示层。逐残基嵌入与 <code>embed</code> 完全相同：fp16 编码器、1024 残基分块、相同的序列清洗，以 fp16 写入
<code>residues.h5</code>（3,460 个数据集，1,102,687 个残基，2.26 GB）。校验：逐残基均值与 <code>embeddings.npz</code> 的最大绝对差为 2.7e-7（fp16 存储），fp32 时为 6e-8。
所有变体都使用第 1 阶段同样的 3 个种子 × 5 折（共 15 个配对折）和同一个内层早停验证集；每个变体族按 <b>seed 0</b> 选出最佳超参，再报告 3 个种子的结果。
池化特征的 MLP 用与第 1 阶段相同的网络族（2×512，LayerNorm，GELU，AdamW，bs 32，早停后在完整训练折上重训），小网格为 dropout {{0.3, 0.5}} × lr {{1e-4, 3e-4}} × z-score。
其中“均值池化”行是在同一协议下的对照，用来把表示层的作用与训练协议的作用分开。
轻注意力把原文 1024 通道的卷积缩窄到 64/128 通道，以满足 1–2M 参数的预算；超过 4,096 个残基的序列（2 条）只取前 4,096 个残基，Transformer 截断到 2,048（17 条）。</p>
{table_html(hdr, trs)}
<img src="{charts['repr']}" alt="表示层逐折配对差">
<details><summary>全部表示层运行（{len(by)} 个配置）</summary>{table_html(['名称', 'AUROC s0/s1/s2', '15 折均值', '参数量'], grid)}</details>
{label_quality_html(best_name)}"""
    return html_out


SEEDS3 = (0, 1, 2)


def label_quality_html(best_name):
    from common import DATA_DIR as D
    f = D / 'pseudogene_caution_pe5.txt'
    if not f.exists():
        return ''
    from sklearn.metrics import roc_auc_score
    caution = {l.strip() for l in open(f) if l.strip()}
    table = load_table()
    acc = np.array([r['accession'] for r in table]); y = np.array([int(r['label']) for r in table])
    flag = np.array([a in caution for a in acc])
    n_c = int((flag & (y == 1)).sum())
    models = [('(b)', 'b_repo_head_sensible'), ('(a)', 'a_repo_defaults'), ('部署头 T1_fixed6', 'T1_fixed6'), ('10 头集成', 'T1_fixed6_ens10')]
    if best_name:
        models.append(('最佳表示层变体 ' + best_name, best_name))
    from common import load_data
    _, _, yy, gg = load_data()
    splits_by_seed = {s: make_splits(yy, gg, s) for s in SEEDS3}
    rows = []
    for lab, name in models:
        preds = [load_oof(name, s) for s in SEEDS3]
        if any(p is None for p in preds):
            continue
        vals = defaultdict(list)
        for s, p in zip(SEEDS3, preds):
            for sp in splits_by_seed[s]:
                te = sp['test']
                yt, pt, ft = y[te], p[te], flag[te]
                for grp, sel in (('caution', ft & (yt == 1)), ('other', ~ft & (yt == 1))):
                    m = (yt == 0) | sel
                    vals[grp + '_auc'].append(roc_auc_score(yt[m], pt[m]))
                    vals[grp + '_recall'].append(float((pt[sel] >= 0.5).mean()))
                vals['all'].append(roc_auc_score(yt, pt))
        rows.append([html.escape(lab)] + [f"{np.mean(vals[k]):.4f}" for k in ('all', 'caution_auc', 'other_auc')] +
                    [f"{np.mean(vals[k]) * 100:.1f}%" for k in ('caution_recall', 'other_recall')])
    return f"""<h3>5.1 标签质量诊断</h3>
<p>从 UniProt REST 查询 PE=5 且带 “Could be the product of a pseudogene” caution 的条目（{len(caution)} 个登录号）；其中 <b>{n_c}</b> 个在本数据集的 1,730 个 PE5 正例里。
下表用折外预测，不重新训练：在每个测试折内，分别以“带 caution 的正例”或“其余正例”对该折全部负例计算 AUROC（召回率按阈值 0.5），再对 15 个折取平均，与正文的指标口径一致。</p>
{table_html(['模型', 'AUROC（全部）', 'AUROC（caution 正例 vs 负例）', 'AUROC（其余 PE5 vs 负例）', '召回率（caution）', '召回率（其余 PE5）'], rows)}"""


def chart_paired_named(rows, fname, title):
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.55 * len(rows)))
    for i, (lab, d) in enumerate(rows):
        y = np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d))
        ax.scatter(d * 100, y, s=18, color=C1, edgecolor=SURF, linewidth=0.5, zorder=3)
        ax.plot([d.mean() * 100] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2, zorder=4)
        ax.annotate(f'{d.mean() * 100:+.2f}', (d.mean() * 100, i - 0.3), xytext=(4, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, color=INK2)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.axvline(2, color=C2, linewidth=1, linestyle='--')
    ax.annotate('目标 +2', (2, len(rows) - 0.4), xytext=(3, 0), textcoords='offset points', fontsize=8, color=C2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel('每折 AUROC 差（百分点；15 个配对折）'); ax.set_title(title)
    return save(fig, fname)


# ------------------------------------------------------------------ html
def table_html(header, rows):
    th = ''.join(f'<th>{h}</th>' for h in header)
    tr = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def main():
    results = read_results()
    table = load_table()
    groups = group_runs(results)
    by_name = {}
    for r in results:
        if r.get('name') and r['seed'] == 0:
            by_name.setdefault(r['name'], r)
    A, B, C = by_name.get('a_repo_defaults'), by_name.get('b_repo_head_sensible'), by_name.get('c_logreg')
    tuned = sorted([r for r in results if r['phase'] == 'baseline_tuned' and r['seed'] == 0], key=lambda r: -r['auroc_mean'])
    Bplus = tuned[0] if tuned else None
    search_rows = [r for r in results if r['phase'] in ('random', 'refine', 'widen', 'custom') and r['seed'] == 0]
    random_rows = [r for r in search_rows if r['phase'] == 'random']
    plain_rows = [r for r in search_rows if r['phase'] in ('random', 'refine')]

    # dataset section
    y = np.array([int(r['label']) for r in table])
    reps = {line.split()[0] for line in open(DATA_DIR / 'clusters_cluster.tsv')}
    from common import load_data
    _, _, yy, gg = load_data()
    sp = make_splits(yy, gg, 0)
    seed = (DATA_DIR / 'seed.txt').read_text().strip()
    charts = {'len': chart_lengths(table)}
    sections = []
    fold_rows = [[k + 1, len(s['test']), int(yy[s['test']].sum()), len(s['test']) - int(yy[s['test']].sum()),
                  len(s['inner_train']), len(s['inner_val'])] for k, s in enumerate(sp)]
    sections.append(f'''<h2>1. 数据集与划分</h2>
<p>正例：Swiss-Prot 全部 PE=5 条目 <b>{int(y.sum())}</b> 条（类 1，"fake"）；负例：从 175,908 条 PE 1–2 中以固定种子 <code>{seed}</code>
均匀抽取 <b>{int((y == 0).sum())}</b> 条（类 0，"real"）。共 {len(y)} 条，MMseqs2 <code>easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0</code>
得到 <b>{len(reps)}</b> 个簇，簇作为分组。评估：StratifiedGroupKFold 5 折（同簇不跨折）；每折训练集再按簇切出 1/8 作内层验证集，
用于早停与 C 选择，测试折从不参与任何选择。嵌入：仓库 <code>ProteinPEClassifier.embed</code>（ProtT5 半精度，1024 残基分块的长度加权均值，GPU）。</p>
{table_html(['折', '测试', '其中 PE5', '其中 PE1–2', '内层训练', '内层验证'], fold_rows)}
<img src="{charts['len']}" alt="序列长度分布">''')

    # baselines
    def row(name, r, desc):
        return [name, desc, f"{r['n_params']:,}"] + [fmt(r[f'{m}_mean'], r[f'{m}_std']) for m, _ in METRICS]
    base_rows = []
    if A: base_rows.append(row('(a) 仓库默认', A, 'Linear(1024,256)-ReLU-Dropout0.1-Linear；Adam lr 1e-3，bs 2，3 epoch，无早停，全训练集'))
    if B: base_rows.append(row('(b) 同结构合理训练', B, '同结构；Adam lr 1e-3，bs 64，≤100 epoch，内层验证 AUROC 早停（patience 15）'))
    if C: base_rows.append(row('(c) 逻辑回归', C, describe(C['config'])))
    if Bplus: base_rows.append(row('(b+) 同结构调参', Bplus, '同结构，仅调 lr/wd/输入 z-score（12 组中最佳）：' + describe(Bplus['config'])))
    sections.append('<h2>2. 基线结果（seed 0，5 折均值 ± 标准差）</h2>' +
                    table_html(['基线', '设置', '参数量'] + [m for _, m in METRICS], base_rows))

    # search
    if search_rows:
        charts['scatter'] = chart_scatter(search_rows, B['auroc_mean'], A['auroc_mean'])
        charts['dw'] = chart_depth_width(plain_rows)
        charts['eff'] = chart_effects(random_rows)
        top = sorted(search_rows, key=lambda r: -r['auroc_mean'])[:10]
        top_rows = [[i + 1, r['phase'], html.escape(describe(r['config'])), f"{r['n_params']:,}", fmt(r['auroc_mean'], r['auroc_std']),
                     fmt(r['mcc_mean'], r['mcc_std'], 3)] for i, r in enumerate(top)]
        counts = defaultdict(int)
        for r in search_rows: counts[r['phase']] += 1
        sections.append(f'''<h2>3. 搜索结果</h2>
<p>共 {len(search_rows)} 个配置（随机 {counts['random']}，细化 {counts['refine']}，扩展 {counts['widen']}），均为 seed 0 的 5 折 CV。
随机空间：层数 {{1,2,3,4}} × 宽度 {{64…1024}} × dropout {{0,0.1,0.3,0.5}} × norm {{none,LN,BN}} × {{ReLU,GELU}} × lr {{1e-4,3e-4,1e-3}}
× wd {{0,1e-4,1e-2}} × bs {{32,64,128}} × label smoothing {{0,0.1}} × 输入 z-score，AdamW，早停 patience 15。</p>
<img src="{charts['scatter']}" alt="AUROC 与参数量"><img src="{charts['dw']}" alt="按层数与宽度">
<img src="{charts['eff']}" alt="超参数效应"><p class="note">箱线图只含随机/细化（效应图只含随机搜索）运行，每点是一个配置的 5 折均值；散点图中集成的参数量按 k 个头累计。注意 seed 0 上的 top-1 带有选择偏差，见第 4 节的多种子复验。</p>
<h3>seed 0 前 10</h3>{table_html(['#', '阶段', '配置', '参数量', 'AUROC', 'MCC'], top_rows)}''')

    # best config with seeds
    names = {}
    for r in results:
        if r.get('name'):
            names.setdefault(ident(r), r['name'])
    multi = [(k, runs, pooled(runs)) for k, runs in groups.items() if pooled(runs) is not None and runs[0]['phase'] != 'repr']
    b_p = pooled(groups[ident(B)]) if B and ident(B) in groups else None
    if multi and b_p is not None:
        a_p = pooled(groups[ident(A)])
        chosen_key = key(json.load(open(CHOSEN))) if CHOSEN.exists() else None
        best = max(multi, key=lambda t: t[2]['auroc'].mean())
        deployed = next((t for t in multi if t[0] == chosen_key), best)
        label_of = lambda k, r0: {'a_repo_defaults': '(a)', 'b_repo_head_sensible': '(b)', 'c_logreg': '(c)'}.get(names.get(k), names.get(k) or r0['phase'])
        seed_rows = []
        for k, runs, p in sorted(multi, key=lambda t: -t[2]['auroc'].mean()):
            r0 = runs[0]
            tag = ('<b>部署</b> ' if k == deployed[0] else '') + ('<b>最佳</b> ' if k == best[0] else '')
            d = p['auroc'] - b_p['auroc']
            seed_rows.append([tag + html.escape(label_of(k, r0)), html.escape(describe(r0['config'])),
                              f"{r0['n_params']:,}" if r0['n_params'] else '—',
                              ' / '.join(f"{runs[s]['auroc_mean']:.4f}" for s in (0, 1, 2)), fmt(p['auroc'].mean(), p['auroc'].std(ddof=1)),
                              f"{d.mean() * 100:+.2f}", f"{int((d > 0).sum())}/15", fmt(p['mcc'].mean(), p['mcc'].std(ddof=1), 3)])

        def comp_rows(t):
            out = []
            for m, lab in METRICS:
                d = t[2][m] - b_p[m]
                out.append([lab, fmt(b_p[m].mean(), b_p[m].std(ddof=1)), fmt(t[2][m].mean(), t[2][m].std(ddof=1)),
                            f'{d.mean() * 100:+.2f} pp', f'{d.mean() / b_p[m].mean() * 100:+.2f}%',
                            f'{int((d > 0).sum())}/{len(d)}', f'{d.std(ddof=1) / np.sqrt(len(d)) * 100:.2f} pp'])
            return out
        hdr = ['指标', '(b) 15 折', '配置 15 折', '绝对提升', '相对提升', '胜出折数', '差值标准误']
        paired = [('部署 − (b)', deployed[2]['auroc'] - b_p['auroc']), ('部署 − (a)', deployed[2]['auroc'] - a_p['auroc'])]
        if best[0] != deployed[0]:
            paired.insert(1, ('最佳 − (b)', best[2]['auroc'] - b_p['auroc']))
        charts['paired'] = chart_paired(paired)
        best_html = '' if best[0] == deployed[0] else f"""<h3>最佳（含集成）配置 vs 基线 (b)：{html.escape(describe(best[1][0]['config']))}</h3>
{table_html(hdr, comp_rows(best))}"""
        sections.append(f'''<h2>4. 最佳配置（3 个种子 × 5 折）</h2>
<p>种子 s ∈ {{0,1,2}} 同时改变 CV 划分（StratifiedGroupKFold 的 random_state）与模型初始化/打乱；同一种子下所有配置用同一划分，因此可逐折配对比较。
下表只列出跑满 3 个种子的配置（seed 0 的前若干名 + 基线 + 扩展变体），按 15 折平均 AUROC 排序；ΔAUROC 为相对 (b) 的逐折配对差均值（百分点）。</p>
{table_html(['配置', '设置', '参数量', 'AUROC s0 / s1 / s2', 'AUROC（15 折）', 'ΔAUROC vs (b)', '胜出折', 'MCC（15 折）'], seed_rows)}
<h3>部署到仓库的配置 vs 基线 (b)：{html.escape(describe(deployed[1][0]['config']))}，参数量 {deployed[1][0]['n_params']:,}</h3>
{table_html(hdr, comp_rows(deployed))}{best_html}
<img src="{charts['paired']}" alt="逐折配对差">''')

    if b_p is not None and any(r['phase'] == 'repr' for r in results):
        sections.append(repr_section(results, groups, b_p, pooled(groups[key(json.load(open(CHOSEN)))]), charts))
    concl = (DOCS / 'conclusion_zh.html').read_text() if (DOCS / 'conclusion_zh.html').exists() else '<p>（搜索进行中，结论待定。）</p>'
    sections.append('<h2>6. 结论</h2>' + concl)
    sections.append('''<h2>7. 复现命令</h2><pre>PY=/root/miniforge3/envs/pp2-pseudo/bin/python
cd scripts/mlp_experiments
$PY prepare_dataset.py            # 平衡子集 + MMseqs2 聚类
$PY embed.py                      # ProtT5 嵌入（GPU 约 2 分钟）
$PY search.py baselines
$PY search.py random --n 300 --rng 1
$PY search.py refine --n 150 --top 10 --rng 2
$PY search.py widen --top 5
$PY search.py custom --configs configs/seeds.json    # 基线与入选配置在 seed 1/2 上复验 + 固定 epoch 变体
$PY search.py custom --configs configs/widen2.json   # RBF-SVM 参照、10 头集成、epoch/宽度/label smoothing 对照
# 表示层实验（第 5 节）
$PY embed_residues.py                       # 逐残基嵌入 → residues.h5（fp16，约 2.3 GB）
$PY repr_search.py features                 # 池化特征缓存
$PY repr_search.py pooled --workers 6
$PY repr_search.py la --workers 2
$PY repr_search.py tf --workers 2
$PY oof_predictions.py                      # 第 1 阶段参照配置的折外预测（标签质量诊断用）
curl -o ../../WebViewer/ModelsAndTrainers/Datasets/mlp_balanced/pseudogene_caution_pe5.txt \\
  'https://rest.uniprot.org/uniprotkb/stream?query=(existence:5)%20AND%20(cc_caution:%22Could%20be%20the%20product%20of%20a%20pseudogene%22)&fields=accession&format=list'
$PY make_report.py</pre><p>细节见 <code>scripts/mlp_experiments/README.md</code>；全部运行记录在 <code>docs/mlp_experiments/results.jsonl</code>。</p>''')

    page = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLP 分类头实验</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e2dd;--accent:#2a78d6}}
body{{background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Noto Sans CJK SC","PingFang SC",sans-serif;max-width:980px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:19px;margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:4px}}h3{{font-size:15px;margin-top:20px}}
p,.note{{color:var(--ink2)}}.note{{font-size:13px}}img{{max-width:100%;display:block;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;display:block;overflow-x:auto}}th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}}
th{{color:var(--ink2);font-weight:600;white-space:nowrap}}td{{font-variant-numeric:tabular-nums}}code,pre{{font-family:ui-monospace,Menlo,monospace;font-size:13px}}
pre{{background:#f3f2ee;padding:10px 12px;overflow-x:auto;border-radius:4px}}
</style></head><body><h1>ProtT5 冻结嵌入上的 MLP 分类头：PE 1–2 vs PE 5</h1>
<p class="note">由 <code>scripts/mlp_experiments/make_report.py</code> 根据 results.jsonl（{len(results)} 条运行记录）自动生成。</p>
{''.join(sections)}</body></html>'''
    (DOCS / 'report.html').write_text(page)
    print(f'wrote {DOCS / "report.html"} ({len(results)} runs)')


if __name__ == '__main__':
    main()
