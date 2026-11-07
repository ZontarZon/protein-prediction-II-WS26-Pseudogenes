"""Embed the balanced subset with the repo's own ProteinPEClassifier.embed (ProtT5, chunked mean)."""
import os
import sys
import time

import numpy as np
import torch

from common import DATA_DIR, WEBVIEWER, load_table

os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
sys.path.insert(0, str(WEBVIEWER))
from ModelsAndTrainers.Models.ProteinPEClassifier import ProteinPEClassifier  # noqa: E402


def read_subset():
    seqs, acc, buf = {}, None, []
    for line in open(DATA_DIR / 'subset.fasta'):
        if line.startswith('>'):
            if acc:
                seqs[acc] = ''.join(buf)
            acc, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    seqs[acc] = ''.join(buf)
    return seqs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seqs = read_subset()
    table = load_table()
    out_path = DATA_DIR / 'embeddings.npz'
    done = dict(np.load(out_path)) if out_path.exists() else {}
    model = ProteinPEClassifier().to(device).eval()
    model.setup_encoder()
    todo = [r['accession'] for r in table if r['accession'] not in done]
    # longest last so progress is visible early; order does not affect the per-protein result
    todo.sort(key=lambda a: len(seqs[a]))
    t0 = time.time()
    for i, acc in enumerate(todo, 1):
        done[acc] = model.embed([seqs[acc]])[0].cpu().numpy().astype(np.float32)
        if i % 250 == 0 or i == len(todo):
            print(f'{i}/{len(todo)} len={len(seqs[acc])} {time.time() - t0:.0f}s', flush=True)
    np.savez(out_path, **done)
    print(f'saved {len(done)} embeddings to {out_path} on {device} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
