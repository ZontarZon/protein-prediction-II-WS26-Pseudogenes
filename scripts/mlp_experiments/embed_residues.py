"""Per-residue ProtT5 embeddings (fp16 HDF5, one dataset per accession).

Same encoder, cleaning and 1024-residue chunking as ProteinPEClassifier.embed: each chunk is
encoded separately and its residue rows (EOS excluded) are concatenated. Checks that the mean
over residues reproduces the cached per-protein embeddings.npz.
"""
import os
import sys
import time

import h5py
import numpy as np
import torch

from common import DATA_DIR, WEBVIEWER, load_table
from embed import read_subset

os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
sys.path.insert(0, str(WEBVIEWER))
from ModelsAndTrainers.Models.ProteinPEClassifier import ProteinPEClassifier, sequence_input  # noqa: E402

OUT = DATA_DIR / 'residues.h5'


def residue_embeddings(model, sequence):
    sequence = sequence_input(sequence)
    device = next(model.head.parameters()).device
    parts = []
    with torch.no_grad():
        for start in range(0, len(sequence), 1024):
            chunk = sequence[start:start + 1024]
            tokens = {k: v.to(device) for k, v in model.tokenizer([' '.join(chunk)], return_tensors='pt', padding=True).items()}
            parts.append(model.encoder(**tokens).last_hidden_state[0, :len(chunk)].float())
    return torch.cat(parts)


def main():
    seqs = read_subset()
    table = load_table()
    model = ProteinPEClassifier().to('cuda').eval()
    model.setup_encoder()
    cached = np.load(DATA_DIR / 'embeddings.npz')
    t0 = time.time()
    worst32 = worst16 = 0.0
    with h5py.File(OUT, 'a') as h5:
        todo = [r['accession'] for r in table if r['accession'] not in h5]
        for i, acc in enumerate(sorted(todo, key=lambda a: len(seqs[a])), 1):
            res = residue_embeddings(model, seqs[acc])
            ref = cached[acc]
            worst32 = max(worst32, float(np.abs(res.mean(0).cpu().numpy() - ref).max()))
            half = res.half().cpu().numpy()
            worst16 = max(worst16, float(np.abs(half.astype(np.float32).mean(0) - ref).max()))
            h5.create_dataset(acc, data=half)
            if i % 500 == 0 or i == len(todo):
                print(f'{i}/{len(todo)} len={len(seqs[acc])} {time.time() - t0:.0f}s '
                      f'max|mean(fp32)-npz|={worst32:.2e} max|mean(fp16)-npz|={worst16:.2e}', flush=True)
        h5.attrs['encoder'] = 'Rostlab/prot_t5_xl_half_uniref50-enc'
        h5.attrs['chunk'] = 1024
    print(f'done {OUT} {OUT.stat().st_size / 1e9:.2f} GB; mean-check fp32 {worst32:.2e}, fp16 storage {worst16:.2e}')


if __name__ == '__main__':
    main()
