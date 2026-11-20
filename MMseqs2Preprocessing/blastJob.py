#!/usr/bin/env python3
"""Stream UniProt JSON into a local BLAST database; search one protein at a time."""
import gzip
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from Bio import SeqIO
import ijson

INPUT_PATH = Path(__file__).with_name('data.fasta')
OUTPUT_DIR = Path(__file__).with_name('outputs')
DATABASE_SOURCE = Path('/mnt/Daten_Programme/Bibliotheken/Downloads/UniProtKB.gz')
DATABASE_DIR = Path(__file__).with_name('local_blast_db')
THREADS = max(1, (os.cpu_count() or 2) - 1)
EVALUE = 10
MAX_HITS = 500


def prepare_database(log):
    """Build once, streaming JSON entries with bounded memory and no FASTA copy."""
    prefix = DATABASE_DIR / 'uniprot'
    marker = DATABASE_DIR / 'complete.json'
    stat = DATABASE_SOURCE.stat()
    signature = [str(DATABASE_SOURCE.resolve()), stat.st_size, stat.st_mtime_ns]
    if marker.exists():
        if json.loads(marker.read_text()) != signature:
            raise ValueError('Database source changed; choose a new DATABASE_DIR to rebuild')
        logging.info('Reusing BLAST database: %s', prefix)
        return prefix
    logging.info('First run: building database from %s. This can take hours and substantial disk space.',
                 DATABASE_SOURCE)
    logging.info('Database: %s; free disk: %.1f GiB. Ctrl+C stops the build; rerun to rebuild.',
                 DATABASE_DIR, shutil.disk_usage(DATABASE_DIR.parent).free / 2**30)
    if sys.stdin.isatty() and input('Build the local database now? [y/N] ').strip().lower() != 'y':
        raise ValueError('Database build cancelled')
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    command = ['makeblastdb', '-dbtype', 'prot', '-title', 'Local UniProtKB',
               '-out', str(prefix), '-in', '-']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                               stderr=subprocess.STDOUT, text=True)
    count = 0
    try:
        with gzip.open(DATABASE_SOURCE, 'rb') as source:
            for entry in ijson.items(source, 'results.item'):
                sequence = entry['sequence']['value']
                accession = entry['primaryAccession']
                name = entry.get('uniProtkbId', accession)
                process.stdin.write(f'>{accession} {name}\n{sequence}\n')
                count += 1
                if count % 100_000 == 0:
                    logging.info('Streamed %s database proteins', f'{count:,}')
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('makeblastdb failed; see blast.log')
        if not count:
            raise ValueError('No proteins found under the JSON results array')
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    marker.write_text(json.dumps(signature) + '\n')
    logging.info('Database ready: %s proteins', f'{count:,}')
    return prefix


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(OUTPUT_DIR / 'blast.log'),
                                  logging.StreamHandler(sys.stdout)], force=True)
    for executable in ('blastp', 'makeblastdb'):
        if not shutil.which(executable):
            raise RuntimeError(f'{executable} missing: install NCBI BLAST+ (Ubuntu/Debian: sudo apt install ncbi-blast+)')
    if THREADS < 1 or EVALUE <= 0 or MAX_HITS < 1:
        raise ValueError('THREADS, EVALUE and MAX_HITS must be positive')
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)
    # Native tool output goes to the same log; Python progress also prints live.
    with (OUTPUT_DIR / 'blast.log').open('a') as log:
        database = prepare_database(log)
        failed = index = 0
        with INPUT_PATH.open() as source:
            for index, record in enumerate(SeqIO.parse(source, 'fasta'), 1):
                name = re.sub(r'[^A-Za-z0-9_.-]', '_', record.id)[:100]
                folder = OUTPUT_DIR / f'{index:04d}_{name}'
                folder.mkdir(parents=True, exist_ok=True)
                query = folder / 'sequence.fasta'
                result = folder / 'result.json'
                temporary = folder / 'result.json.tmp'
                SeqIO.write(record, query, 'fasta')
                # Do not leave an old success beside a new failed query.
                result.unlink(missing_ok=True)
                sequence = str(record.seq).upper()
                if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYBXZJUO'):
                    logging.error('%s: empty sequence or invalid amino-acid characters', record.id)
                    failed += 1
                    continue
                logging.info('[%s] Searching %s (%s residues), %s threads; Ctrl+C to stop',
                             index, record.id, len(sequence), THREADS)
                subprocess.run(['blastp', '-query', str(query), '-db', str(database),
                                '-num_threads', str(THREADS), '-evalue', str(EVALUE),
                                '-max_target_seqs', str(MAX_HITS), '-outfmt', '15',
                                '-out', str(temporary)], stdout=log, stderr=subprocess.STDOUT,
                               check=True)
                with temporary.open() as handle:
                    json.load(handle)
                temporary.replace(result)
                logging.info('[%s] Saved %s', index, result)
    logging.info('Processed %s proteins; %s invalid. Results: %s', index, failed, OUTPUT_DIR)
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.info('Stopped. Completed results remain; rerunning searches all queries again.')
        raise SystemExit(130)
    except Exception:
        logging.exception('BLAST run failed; see blast.log')
        raise SystemExit(1)
