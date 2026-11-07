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
