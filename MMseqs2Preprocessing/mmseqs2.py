"""Cluster the paired FASTA and export entries with ProtT5 embeddings."""

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from xmlProcess import build_pairs

ROOT = Path(__file__).resolve().parent
INPUT_FASTA = ROOT / "paired_hits.fasta"
PSEUDOGENE_FASTA = ROOT / "data.fasta"
BLAST_DIR = ROOT / "blast_results"
OUTPUT_DIR = ROOT / "outputs"
MMSEQS = "mmseqs"
THREADS = 4
MIN_IDENTITY = 0.5
COVERAGE = 0.8
COVERAGE_MODE = 0
CHECKPOINT = "Rostlab/prot_t5_xl_half_uniref50-enc"
CHUNK_SIZE = 1024


def embed_entries(entries):
    os.environ.setdefault("HF_HOME", str(OUTPUT_DIR / ".huggingface"))
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(THREADS)
    print(f"Loading {CHECKPOINT} on {device}", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT, do_lower_case=False)
    dtype = torch.float16 if device == "cuda" else torch.float32
    encoder = T5EncoderModel.from_pretrained(CHECKPOINT, torch_dtype=dtype).to(device).eval()

    vectors = {}
    with torch.inference_mode():
        for entry in entries.values():
            sequence = re.sub(r"[UZOBJ]", "X", entry["sequence"].upper())
            if not sequence or re.search(r"[^ACDEFGHIKLMNPQRSTVWYX]", sequence):
                raise ValueError(f"Invalid protein sequence in FASTA record {entry['id']}")
            if sequence not in vectors:
                total = torch.zeros(encoder.config.d_model, device=device, dtype=torch.float32)
                for start in range(0, len(sequence), CHUNK_SIZE):
                    chunk = sequence[start:start + CHUNK_SIZE]
                    tokens = tokenizer(" ".join(chunk), return_tensors="pt").to(device)
                    hidden = encoder(**tokens).last_hidden_state
                    total += hidden[0, :len(chunk)].float().sum(0)
                vectors[sequence] = (total / len(sequence)).cpu().tolist()
            entry["embedding"] = vectors[sequence]
            if entry["id"] % 100 == 0:
                print(f"Embedded {entry['id']}/{len(entries)} records", flush=True)


def main():
    executable = shutil.which(MMSEQS)
    if executable is None:
        raise SystemExit("Install MMseqs2 or set MMSEQS to the path of its executable.")

    pseudogene_accessions = build_pairs(PSEUDOGENE_FASTA, BLAST_DIR, INPUT_FASTA)

    with TemporaryDirectory(prefix="mmseqs2-") as directory:
        work = Path(directory)
        numbered_fasta = work / "records.fasta"
        entries = {}
        # A UniProt ID can occur twice, so use the 1-based FASTA position.
        fasta = INPUT_FASTA.read_text().strip()
        if not fasta:
            raise ValueError(f"No FASTA records in {INPUT_FASTA}")
        records = fasta.removeprefix(">").split("\n>")
        with numbered_fasta.open("w") as numbered:
            for record_id, record in enumerate(records, start=1):
                header, _, sequence_lines = record.partition("\n")
                fasta_id, *description = header.split(maxsplit=1)
                database, accession, entry_name = fasta_id.split("|")
                parts = re.split(r"\s+(OS|OX|GN|PE|SV)=", description[0] if description else "")
                annotations = dict(zip(parts[1::2], parts[2::2]))
                sequence = "".join(sequence_lines.split())
                entries[record_id] = {
                    "id": record_id,
                    "fasta_id": fasta_id,
                    "header": header,
                    "database": database,
                    "accession": accession,
                    "tag": "pseudogene" if accession in pseudogene_accessions else "regular",
                    "entry_name": entry_name,
                    "protein_name": parts[0],
                    "organism": annotations.get("OS"),
                    "taxonomy_id": annotations.get("OX"),
                    "gene_name": annotations.get("GN"),
                    "protein_existence": annotations.get("PE"),
                    "sequence_version": annotations.get("SV"),
                    "sequence": sequence,
                }
                numbered.write(f">{record_id}\n{sequence}\n")

        prefix = work / "clusters"
        subprocess.run(
            [
                executable, "easy-cluster", str(numbered_fasta), str(prefix),
                str(work / "tmp"), "--dbtype", "1", "--threads", str(THREADS),
                "--min-seq-id", str(MIN_IDENTITY), "-c", str(COVERAGE),
                "--cov-mode", str(COVERAGE_MODE),
            ],
            check=True,
        )

        clusters = {}
        seen = set()
        with (work / "clusters_cluster.tsv").open(newline="") as result:
            for representative, member in csv.reader(result, delimiter="\t"):
                cluster_id, member_id = int(representative), int(member)
                if member_id in seen or cluster_id not in entries:
                    raise ValueError("Invalid cluster membership")
                clusters.setdefault(cluster_id, []).append(entries[member_id])
                seen.add(member_id)
        if seen != entries.keys():
            raise ValueError("Expected exactly one cluster membership per FASTA record")

        embed_entries(entries)
        output = [
            {"cluster_id": cluster_id, "items": items}
            for cluster_id, items in clusters.items()
        ]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=OUTPUT_DIR) as staging:
            pending = Path(staging) / "clusters.json"
            with pending.open("w", encoding="utf-8") as file:
                json.dump(output, file, ensure_ascii=False, allow_nan=False)
                file.write("\n")
            pending.replace(OUTPUT_DIR / "clusters.json")
        shutil.copyfile(work / "clusters_rep_seq.fasta", OUTPUT_DIR / "cluster_representatives.fasta")
        shutil.copyfile(work / "clusters_all_seqs.fasta", OUTPUT_DIR / "cluster_sequences.fasta")

    print(f"Wrote {len(clusters)} clusters with {len(entries)} records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
