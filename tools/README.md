# tools

`download_datasets.py` fetches the project datasets from Google Drive using `datasets_manifest.json`. Large files are stored as 7 MB parts that get concatenated back together. Every part and every rebuilt file is checked with SHA-256, and files that already match are skipped, so you can safely re-run it. Output goes to `WebViewer/ModelsAndTrainers/Datasets/`. Only the Python 3.9+ standard library is needed.

    python tools/download_datasets.py                       # everything (~191 MB)
    python tools/download_datasets.py --only mlp_balanced   # one dataset folder
    python tools/download_datasets.py --only prepared.pt uniprot_pseudogene_caution
    python tools/download_datasets.py --list                # show manifest, no download

Other options: `--root PATH`, `--keep-parts`, `--manifest PATH`, `--gdown` (uses `gdown` if it is installed).

Drive folder: https://drive.google.com/drive/folders/16ZLjGsKJia4ZorFjWD41218pLLuxq1T9
The folder must be shared with the downloader's Google account or set to "anyone with the link". Otherwise the script stops with a sign-in error.

- `uniprot_full_fasta/`: raw Swiss-Prot 2026_03 (`uniprot_sprot.fasta.gz`) plus the pseudogene caution-comment TSV.
- `mmseqs2_prepared/`: `prepared.pt`, the full imbalanced dataset with its cluster-wise 80/20 train/validation split (label 0 = PE1/2, label 1 = PE5).
- `mlp_balanced/`: 1:1 subset of PE5 vs PE1/2, with MMseqs2 clusters, seed, and ProtT5 `embeddings.npz`.
