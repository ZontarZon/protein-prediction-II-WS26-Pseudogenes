"""Build the class-balanced PE subset and its MMseqs2 clusters.

Positives: every Swiss-Prot PE=5 entry. Negatives: an equal number of PE 1-2
entries sampled uniformly (random.Random(SEED).sample over file order).
"""
import argparse
import gzip
import random
import re
import shutil
import subprocess
import sys

from common import DATA_DIR, SPROT

SEED = 20261106
MMSEQS = '/root/miniforge3/envs/pp2-pseudo/bin/mmseqs'


def read_fasta(path):
    header, seq = None, []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line.strip())
    if header is not None:
        yield header, ''.join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pos, neg = [], []
    for header, seq in read_fasta(SPROT):
        labels = re.findall(r'(?<!\S)PE=([^\s]*)', header)  # same PE-field rule as MMseqs2ClassificationData.load_records
        if len(labels) != 1:
            raise ValueError(header)
        pe = int(labels[0])
        acc = header.split('|')[1]
        if pe == 5:
            pos.append((acc, pe, header, seq))
        elif pe in (1, 2):
            neg.append((acc, pe, header, seq))
    rng = random.Random(args.seed)
    neg_sel = rng.sample(neg, len(pos))
    chosen = [(r, 1) for r in pos] + [(r, 0) for r in neg_sel]
    print(f'PE5={len(pos)} PE1-2 available={len(neg)} sampled={len(neg_sel)} seed={args.seed}')
    with open(DATA_DIR / 'subset.fasta', 'w') as fa, open(DATA_DIR / 'subset.tsv', 'w') as tsv:
        tsv.write('accession\tpe\tlabel\tlength\n')
        for (acc, pe, header, seq), label in chosen:
            fa.write(f'>{acc} {header}\n')
            for i in range(0, len(seq), 60):
                fa.write(seq[i:i + 60] + '\n')
            tsv.write(f'{acc}\t{pe}\t{label}\t{len(seq)}\n')
    (DATA_DIR / 'seed.txt').write_text(f'{args.seed}\n')
    tmp = DATA_DIR / 'mmseqs_tmp'
    cmd = [MMSEQS, 'easy-cluster', str(DATA_DIR / 'subset.fasta'), str(DATA_DIR / 'clusters'), str(tmp),
           '--min-seq-id', '0.3', '-c', '0.8', '--cov-mode', '0', '--threads', str(args.threads)]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True, stdout=open(DATA_DIR / 'mmseqs.log', 'w'), stderr=subprocess.STDOUT)
    shutil.rmtree(tmp, ignore_errors=True)
    reps = {line.split()[0] for line in open(DATA_DIR / 'clusters_cluster.tsv')}
    print(f'clusters={len(reps)}')


if __name__ == '__main__':
    sys.exit(main())
