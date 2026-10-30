# protein-prediction-II-WS26-Pseudogenes
Pseudogene project for the TUM course, Protein Prediction II.

## Repository layout

```text
protein-prediction-II-WS26-Pseudogenes/
├── dataset/                   one folder per source; files stay local
│   ├── uniprot_pseudogene_candidates/   Swiss-Prot entries with the pseudogene caution
│   ├── deleted_swissprot/               optional removed Swiss-Prot entries
│   ├── gencode_pseudogenes/             optional human GENCODE pseudogenes
│   └── functional_proteins/             PE1/PE2 proteins without that caution
├── docs/                      reports and slides (placeholder)
├── models/                    model definitions and training code (placeholder)
├── scripts/                   download and preprocessing scripts (placeholder)
├── .gitignore                 dataset file contents stay out of git
├── AGENTS.md                  commit message convention
└── README.md                  project overview and data sources
```

`dataset/` is the only data directory. Each subdirectory is one source, because the sequences, labels, and embeddings do not come from the same place. The folders are part of the layout; the files inside them are gitignored.

`docs/`, `models/`, and `scripts/` are empty for now. Every empty folder holds a `.gitkeep` so that it survives a clone.

## Datasets and ProtT5 embeddings

| Dataset | Description | ProtT5 embedding download |
|---|---|---|
| [UniProt pseudogene candidates](dataset/uniprot_pseudogene_candidates) | Swiss-Prot entries with the caution "Could be the product of a pseudogene". The course query `(existence:5) AND (cc_caution:"Could be the product of a pseudogene")` returns 811 entries; we keep all 820 reviewed entries with this caution (9 of them are PE1-3). Sequences are stop-free (hard cases). | [Swiss-Prot per-protein.h5](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/uniprot_sprot/per-protein.h5) (1.3 GB, release 2026_03; select entries by accession) |
| [Deleted Swiss-Prot entries](dataset/deleted_swissprot) (optional) | Entries listed in [`delac_sp.txt`](https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/docs/) whose last version in [UniSave](https://rest.uniprot.org/unisave/api/docs) carries the pseudogene caution. | No precomputed embeddings: deleted entries are not in the current Swiss-Prot file, and UniProt does not archive embeddings in previous releases. Compute with the [ProtSpace notebook](https://github.com/tsenoner/protspace/blob/main/apps/protspace/notebooks/ClickThrough_GenerateEmbeddings.ipynb) or the [biocentral API](https://github.com/biocentral/biocentral_api). |
| [GENCODE pseudogenes](dataset/gencode_pseudogenes) (optional) | Human pseudogenes annotated in the [GENCODE](https://www.gencodegenes.org/human/) GTF. GENCODE provides no translations; build them yourself, e.g. from the parent protein (the "2-way consensus pseudogenes" file lists parents for about 8,900). Contain stops and frameshifts. | No precomputed embeddings: the translations are self-made. Compute with the [ProtSpace notebook](https://github.com/tsenoner/protspace/blob/main/apps/protspace/notebooks/ClickThrough_GenerateEmbeddings.ipynb) or the [biocentral API](https://github.com/biocentral/biocentral_api). The [human reference proteome per-protein.h5](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5) (356 MB) covers the human parent proteins only. |
| [Functional proteins](dataset/functional_proteins) | Reviewed Swiss-Prot entries with evidence at protein or transcript level (PE1/PE2) and no pseudogene caution. | [Swiss-Prot per-protein.h5](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/uniprot_sprot/per-protein.h5) (same file as the candidates) |

Notes:

- The UniProt files hold one vector per protein; sequences longer than 12,000 residues are excluded ([README](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/README)).
- Per-residue embeddings for Swiss-Prot are at [ftp.ebi.ac.uk/pub/contrib/UniProt/embeddings/current_release/uniprot_sprot](https://ftp.ebi.ac.uk/pub/contrib/UniProt/embeddings/current_release/uniprot_sprot/) (398 GB). That file is dated 2026-07-22, before release 2026_03, so check that the accessions match.
- Neither ProtT5 nor ESM-2 has a token for `*`: decide how stops and frameshifts are represented before embedding self-made translations.
