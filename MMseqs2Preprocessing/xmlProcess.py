"""Pair each query with its highest-scoring BLAST hit outside the query set."""

from pathlib import Path
import re
from textwrap import fill
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
QUERY_FASTA = ROOT / "data.fasta"
BLAST_DIR = ROOT / "blast_results"
OUTPUT_FASTA = ROOT / "paired_hits.fasta"
LINE_WIDTH = 60

NS = {"": "http://www.ebi.ac.uk/schema"}
EXCLUDED_PE = re.compile(r"(?:^|\s)PE=[345](?=\s|$)")


def build_pairs(query_fasta, blast_dir, output_fasta):
    queries = query_fasta.read_text().strip().removeprefix(">").split("\n>")
    query_accessions = {query.split(None, 1)[0].split("|")[1] for query in queries}
    hits = {}
    partial = no_hits = 0
    for xml_file in sorted(blast_dir.glob("*.xml")):
        result = ElementTree.parse(xml_file)
        query = result.find("Header/parameters/sequences/sequence", NS).attrib["name"]
        candidates = (
            (hit, alignment)
            for hit in result.findall("SequenceSimilaritySearchResult/hits/hit", NS)
            if hit.attrib["ac"] not in query_accessions
            and not EXCLUDED_PE.search(hit.get("description", ""))
            for alignment in hit.findall("alignments/alignment", NS)
        )
        hit, alignment = max(
            candidates, key=lambda pair: int(pair[1].findtext("score", namespaces=NS)),
            default=(None, None),
        )
        if query in hits:
            raise ValueError(f"Duplicate BLAST result for {query}")
        if hit is None:
            hits[query] = ""
            no_hits += 1
            print(f"No eligible BLAST hits for {query}")
            continue

        match = alignment.find("matchSeq", NS)
        sequence = match.text.replace("-", "")
        fields = hit.attrib
        hit_id = f"{fields['database'].lower()}|{fields['ac']}|{fields['id']}"
        header = f">{hit_id} {fields['description']}".rstrip()
        hits[query] = f"{header}\n{fill(sequence, width=LINE_WIDTH)}\n"
        partial += len(sequence) < int(fields["length"])

    records = []
    for query in queries:
        query_id = query.split(None, 1)[0]
        if query_id not in hits:
            raise ValueError(f"Missing BLAST result for {query_id}")
        if hit_record := hits.pop(query_id):
            records.extend((f">{query.rstrip()}\n", hit_record))
    if hits:
        raise ValueError(f"BLAST result without a FASTA query: {next(iter(hits))}")
    output_fasta.write_text("".join(records))
    print(f"Wrote {len(records) // 2} pairs ({partial} partial hits, {no_hits} unpaired queries skipped)", flush=True)
    return query_accessions


if __name__ == "__main__":
    build_pairs(QUERY_FASTA, BLAST_DIR, OUTPUT_FASTA)
