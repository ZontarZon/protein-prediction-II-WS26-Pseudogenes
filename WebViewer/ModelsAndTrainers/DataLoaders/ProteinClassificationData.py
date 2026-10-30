"""PE-labelled protein classification using the shared streaming FASTA loader."""
import re

from ModelsAndTrainers.DataLoaders.FastaDataLoader import FastaDataset


class ProteinClassificationDataset(FastaDataset):
    """Stream FASTA entries and map PE 1–4 to true (0), PE 5 to false (1).

    Inherits plain/gzip file discovery and worker sharding from FastaDataset.
    Sequences and headers stay available; only sequences go into the model.
    """

    def __iter__(self):
        for record in super().__iter__():
            # Parse the entire field so malformed values such as PE=10 or a
            # second PE field cannot silently become a valid training target.
            labels = re.findall(r'(?<!\S)PE=([^\s]*)', record['header'])
            if len(labels) != 1 or labels[0] not in {'1', '2', '3', '4', '5'}:
                raise ValueError(f"{record['id']}: expected one PE=1..5 header label")
            pe = int(labels[0])
            yield {**record, 'pe': pe, 'target': int(pe == 5)}


def records(root):
    """Compatibility iterator for callers using the original reader function."""
    yield from ProteinClassificationDataset(root)
