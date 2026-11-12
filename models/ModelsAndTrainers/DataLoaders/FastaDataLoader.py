"""Stream protein FASTA records without loading a complete file into memory."""
import gzip
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info


_FASTA_SUFFIXES = ('.fasta', '.fa', '.faa', '.fas')
# Includes ambiguous/rare residues, stop markers, and alignment gaps.
_PROTEIN_SYMBOLS = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZ*-')


def read_fasta(path, max_record_chars=None):
    """Yield dictionaries with id, full header (without >), sequence, and length.

    Blank lines and semicolon comments are ignored. Multiline sequences are
    joined, whitespace removed, and letters uppercased. Residues are preserved
    (including B/J/O/U/X/Z); model-specific replacement belongs in tokenization.
    Malformed or empty records raise ValueError with a filename and line number.
    Only the current record is buffered. Plain text and gzip files are supported.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    header = None
    header_line = 0
    sequence_parts = []
    record_chars = 0

    def record():
        if not sequence_parts:
            raise ValueError(f'{path}:{header_line}: FASTA record has no sequence')
        sequence = ''.join(sequence_parts)
        return {'id': header.split()[0], 'header': header,
                'sequence': sequence, 'length': len(sequence)}

    with opener(path, 'rt', encoding='utf-8-sig') as source:
        lines = source if max_record_chars is None else iter(lambda: source.readline(max_record_chars + 2), '')
        for line_number, raw in enumerate(lines, 1):
            if max_record_chars is not None and len(raw) > max_record_chars + 1:
                raise ValueError(f"{path}:{line_number}: Line exceeds the preview size limit")
            line = raw.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('>'):
                if header is not None:
                    yield record()
                header = line[1:].strip()
                header_line = line_number
                sequence_parts = []
                record_chars = 0
                if not header:
                    raise ValueError(f'{path}:{line_number}: FASTA header is empty')
            else:
                if header is None:
                    raise ValueError(f'{path}:{line_number}: Sequence appears before a FASTA header')
                sequence = ''.join(line.split()).upper()
                invalid = set(sequence) - _PROTEIN_SYMBOLS
                if invalid:
                    raise ValueError(f'{path}:{line_number}: Invalid protein symbols: {sorted(invalid)}')
                record_chars += len(sequence)
                if max_record_chars is not None and record_chars > max_record_chars:
                    raise ValueError(f"{path}:{line_number}: Protein exceeds the preview size limit")
                sequence_parts.append(sequence)
        if header is not None:
            yield record()


class FastaDataset(IterableDataset):
    """Re-iterable, streaming PyTorch dataset for a FASTA file or directory.

    Directories are searched recursively for common FASTA extensions and their
    .gz variants. Construction discovers filenames only; contents are opened
    during iteration. DataLoader workers receive disjoint sets of files; a
    single input file therefore uses just one worker. There is no random access,
    length scan, or shuffle support. Default PyTorch collation batches the raw
    strings; apply a pLM tokenizer to each batch's ``sequence`` list separately.
    """

    def __init__(self, root_dir):
        super().__init__()
        root = Path(root_dir)
        if root.is_file():
            self.paths = (root,)
        elif root.is_dir():
            self.paths = tuple(sorted(
                path for path in root.rglob('*')
                if path.is_file() and path.name.lower().endswith(
                    _FASTA_SUFFIXES + tuple(suffix + '.gz' for suffix in _FASTA_SUFFIXES)
                )
            ))
            if not self.paths:
                raise ValueError(f'No FASTA files found under {root}')
        else:
            raise FileNotFoundError(root)

    def __iter__(self):
        worker = get_worker_info()
        paths = self.paths if worker is None else self.paths[worker.id::worker.num_workers]
        for path in paths:
            yield from read_fasta(path)


# Opt-in backend preview contract. Detection reads this declaration without
# importing the loader; only a preview request opens the selected file.
PREVIEW_EXTENSIONS = ('.fasta', '.fa', '.faa', '.fas', '.fasta.gz', '.fa.gz', '.faa.gz', '.fas.gz')


def preview_records(path):
    return read_fasta(path, max_record_chars=1_000_000)
