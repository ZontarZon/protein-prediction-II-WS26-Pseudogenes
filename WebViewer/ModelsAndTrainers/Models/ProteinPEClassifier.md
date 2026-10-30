# Frozen ProtT5 PE classification

The model uses the reference notebook's [Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc). This is a **protein** encoder. Inputs are amino-acid sequences, not nucleotide DNA. Each residue is spaced for tokenization; U/Z/O/B/J become X. Mean pooling excludes padding and EOS. The frozen encoder always stays in evaluation mode, with no gradients; a 1024 → 256 → 2 head learns the PE grouping with cross entropy. Softmax produces the two displayed probabilities.

Install dependencies with `python -m pip install -r backend/requirements.txt` in your training environment. The first actual embedding request downloads the large pretrained checkpoint; catalog discovery and imports do not download it. CUDA uses half-precision encoder weights and CPU uses float32; CPU operation requires substantial RAM and is slow. The head uses float32.

In **New run**, select `ProteinPETrainer`, `ProteinPEClassifier` for its classifier role, and a PE-labelled dataset. The trainer uses `ProteinClassificationDataset`, which extends the shared streaming `FastaDataset`. Select a single FASTA file or a directory of `.fasta`, `.fa`, `.faa`, `.fas`, and their gzip variants. JSON is no longer part of this training path. Headers must contain exactly one `PE=1` through `PE=5`. Missing/invalid labels fail explicitly. Class 1 is PE 1–4; class 2 is PE 5. These are evidence classes, not proof of authenticity.

The complete dataset is indexed once per run into `protein_samples.sqlite` in the run directory. This disk cache stores normalized sequences; record IDs and sequence-group digests are held in memory for a complete random permutation of each split every epoch. There is no maximum-protein-length filter or configurable shuffle buffer. Optional per-class subset counts control how many records are used. Allow disk space for a copy of the sequence data plus SQLite overhead. A fresh random selection seed is generated for each run. `subset_real_count` and `subset_fake_count` independently cap their classes; 0 uses the entire class. Sampling is uniform across the full class, not a prefix. Selection happens once before the validation split and remains fixed across epochs. Checkpoints preserve the seed, subset counts, validation percentage, and a fingerprint of the ordered dataset, so resume reconstructs the same records and refuses a changed dataset. Identical normalized sequences stay together in the split; duplicate groups may shift the requested `validation_percent` slightly. Related proteins can still cross the split. Both classes must occur in both partitions.

All residues contribute to the protein embedding. The frozen encoder processes consecutive chunks of at most 1,024 residues, then combines their residue sums into a length-weighted mean. This avoids a length-dependent skip or truncation and bounds attention memory, but residues in separate chunks do not share encoder context. Batch size controls head-training batches; encoder chunks are processed sequentially.

Training and validation run as separate phases. Metrics flow through the same backend report mechanism as `DiffusionTrainer`: phase, batch counts, epoch position, loss, accuracy, learning rate, elapsed time, and class counts. Training progress is per epoch. Validation has an orange progress bar that resets after evaluation. The web view overlays orange validation loss on training loss and also shows validation in its own graph. Model download/loading progress bars are disabled; warnings and errors remain visible. No automatic class balancing or probability calibration is applied.

Each completed epoch saves the head, architecture, optimizer, progress, and CPU RNG state. Encoder weights are referenced by Hugging Face checkpoint name rather than copied into every checkpoint. Resume occurs at completed epoch boundaries; a stopped partial epoch may be replayed. CUDA RNG state is not restored, so resume is not bit-for-bit reproducible on GPU. For reproducible publication runs, pin the Hugging Face revision and dependency versions in the environment.

In **Inference → New inference**, choose **ProtT5 PE classifier**, select a trained checkpoint, choose a device, and paste one raw protein sequence or one full FASTA entry. Header labels are never passed to the model. Results show Class 1 (Real value), Class 2 (Fake value), and a circular probability meter. These are uncalibrated model probabilities. No trained task head is supplied; train one before inference.

## First test run with the replacement FASTA

In **New run**, use:

| Setting | Value |
| --- | --- |
| Trainer | `ProteinPETrainer` |
| Classifier | `ProteinPEClassifier` |
| Training data | `uniprot_full_fasta` |
| Epochs | `1` |
| Batch size | `2` (use `1` if GPU memory is tight) |
| Validation percentage | `20` |
| Learning rate | `0.001` |
| Device | A CUDA device if available |

By default every FASTA entry is used in training or validation, including long proteins. Set each class’s Subset count in the dataset preview to limit that class before splitting; for example, 100 Real and 100 Fake gives about 160 training and 40 validation elements at 20% validation. The trainer indexes the complete dataset and checks class coverage before loading ProtT5. Start with one epoch to check the setup. The full dataset must be scanned to draw a uniform subset; model training then uses only the selected elements. Old `max_records`, `max_residues`, and `shuffle_buffer` trainer settings are ignored.

After one completed epoch, select its `checkpoint_0001.pt` from the inference page and paste a protein sequence or one FASTA entry. A full-model test requires the pretrained download and enough device memory; automated tests use a small stand-in encoder to check training, saving, resume, and inference without that download.

## Dataset and batch previews

Selecting a dataset starts an asynchronous, model-free count of its records. The setup view shows the total and counts for Real (PE 1–4) and Fake (PE 5), independent subset inputs, and estimated training/validation batches per epoch. Counts are cached by file paths, sizes, and modification times until backend restart or file changes. Batch estimates update immediately with subset counts, validation percentage, and batch size; partial batches count as one. Exact totals are reported during training after duplicate-safe splitting.

Subset counts above availability use all available entries. Small subsets need at least two distinct sequences per class to populate both train and validation. Checkpoints produced before per-run subset selection do not contain the required selection state; start a new run for this workflow.
