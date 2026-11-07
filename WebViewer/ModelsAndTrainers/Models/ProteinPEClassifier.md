# ProtT5 pseudogene-caution classifier

This model uses the protein encoder `Rostlab/prot_t5_xl_half_uniref50-enc` with
frozen weights and a trainable binary head. Inputs are amino-acid sequences;
the model mean-pools residue embeddings in chunks of up to 1,024 residues.

Prepare data with [MMseqs2 preprocessing](../../../MMseq2Preprocessing/README.md).
That workflow joins a Swiss-Prot FASTA to a UniProt caution TSV, labels PE5
entries with the pseudogene caution as class 0, and PE1/PE2 entries without
that caution as class 1. MMseqs2 clusters sequences so related records remain
in the same train or validation split. Select `PLM_MLPTrainer` and this model,
then choose the prepared `.pt` dataset. MMseqs2 is needed for preprocessing,
not for training or inference.

The first embedding run downloads the pretrained encoder. CUDA uses
half-precision encoder weights; CPU uses float32 and requires substantial RAM.
Checkpoints store the trained head and refer to the pretrained encoder by its
Hugging Face identifier.

Inference accepts one protein sequence or FASTA entry. The two output
probabilities correspond to PE5 with the pseudogene caution and PE1/PE2 without
the caution. They are model scores, not calibrated certainty.

## Classification head

The head is `[Linear → norm → activation → Dropout] × len(hidden_sizes) → Linear(…, 2)` on the 1024-d protein embedding. Model parameters:

| Parameter | Default | Options |
| --- | --- | --- |
| `hidden_sizes` | `[512, 512]` | list of positive widths |
| `dropout` | `0.5` | `[0, 1)` |
| `norm` | `layernorm` | `none`, `layernorm`, `batchnorm` |
| `activation` | `gelu` | `relu`, `gelu` |

The default (1024 → 512 → 512 → 2, 790,530 parameters) was selected by a homology-grouped 5-fold CV search over about 570 runs on a class-balanced Swiss-Prot subset (all 1,730 PE 5 entries + 1,730 random PE 1–2 entries, MMseqs2 clusters at 30% identity); see `docs/mlp_experiments/report.html` and `scripts/mlp_experiments/`. That search predates the pseudogene-caution filter: it used every PE 5 entry, not only those with the caution. Its gain over the old 1024 → 256 → 2 head is small: about +0.5 AUROC points against the old head trained with batch size 64 and early stopping, and +0.26 against the old trainer defaults (0.956 vs. 0.951–0.954 AUROC over 3 seeds × 5 folds). Logistic regression and an RBF-SVM on the same embeddings reach 0.951–0.955, so the frozen mean embedding, not the head, limits accuracy. With `batchnorm`, a final training batch of one sequence fails, so avoid batch sizes that leave a remainder of 1.

`PLM_MLPTrainer` optimises the head with AdamW (defaults: learning rate 1e-4, weight decay 0, batch size 32, 6 epochs). These defaults train for about 520 optimizer steps on the balanced 2.8k-sequence training folds. Larger datasets give more steps per epoch, so lower the epoch count to stay near that budget. The trainer has no early stopping and does not cache embeddings; every epoch re-encodes its sequences.

Checkpoints store the head arguments in `model_settings` (`hidden_sizes`, `dropout`, `norm`, `activation`). **Old checkpoints (`model_settings = {'hidden_size': 256}`) are incompatible with the new default head**: you cannot import them as initial weights for the default model or resume them with it. Inference still works because `infer` rebuilds the head from the checkpoint's own `model_settings`, and the legacy `hidden_size` argument reproduces the old 1024 → hidden_size → 2 layout (ReLU, dropout 0.1). Those checkpoints were trained with the earlier label order (class 0 = PE 1–2, class 1 = PE 5), so their two displayed probabilities are swapped relative to the current labels. To resume an old run, set the model parameter `hidden_size=256`.
