#!/usr/bin/env python3
"""Run sequential BLASTP searches: load FASTA, submit, wait, download, repeat.

Edit the settings below, then run `python MMseqs2Preprocessing/blastJob.py`.
Each protein gets its own job. Batches only group the local input records;
there is never more than one submitted job being processed at a time.
"""
from itertools import batched
import logging
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from Bio import SeqIO
import requests
from tqdm import tqdm

# Settings: paths are relative to this script; all delays are in seconds.
INPUT_PATH = Path(__file__).with_name('data.fasta')
OUTPUT_DIR = Path(__file__).with_name('blast_results')
EMAIL = 'jeyemo3494@jcaton.com'  # Replace with your real contact email before running.
BATCH_SIZE = 5          # Number of FASTA records loaded at once (not concurrent jobs).
POLL_INTERVAL = 30      # Wait before the first status check AND every later check.
SUBMISSION_DELAY = 10   # Pause after each job before submitting the next protein.
REQUEST_TIMEOUT = 120    # Maximum wait for an individual HTTP request.
STATUS_RETRIES = 3      # Retry throttled status checks, never job submissions.
RETRY_DELAY = 30        # Backoff starts here, then doubles for each retry.
BASE_URL = 'https://www.ebi.ac.uk/Tools/services/rest/ncbiblast'


def request(session, method, endpoint, **kwargs):
    """Log EBI's error body; retry only rate-limited status checks."""
    for attempt in range(STATUS_RETRIES + 1):
        response = session.request(method, BASE_URL + endpoint, timeout=REQUEST_TIMEOUT, **kwargs)
        if response.ok:
            return response
        # EBI sometimes wraps its scheduler's HTTP 429 inside an HTTP 400.
        throttled = response.status_code == 429 or (
            response.status_code == 400 and 'rate_limit_exceeded' in response.text
        )
        if method == 'GET' and endpoint.startswith('/status/') and throttled and attempt < STATUS_RETRIES:
            delay = max(POLL_INTERVAL, RETRY_DELAY * 2 ** attempt)
            retry_after = response.headers.get('Retry-After', '')
            if retry_after.isdigit():
                delay = max(delay, int(retry_after))
            logging.warning('%s %s: HTTP %s: %s; retry in %ss',
                            method, endpoint, response.status_code, response.text.strip(), delay)
            time.sleep(delay)
            continue
        logging.error('%s %s: HTTP %s: %s',
                      method, endpoint, response.status_code, response.text.strip())
        response.raise_for_status()


def search(session, record, destination):
    """Submit one protein, poll with pauses, and save its XML result."""
    sequence = str(record.seq).upper()
    if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYBXZJUO'):
        raise ValueError('Empty sequence or invalid amino-acid characters')
    job_id = request(session, 'POST', '/run', data={
        'email': EMAIL, 'program': 'blastp', 'database': 'uniprotkb',
        'stype': 'protein', 'sequence': f'>{record.id}\n{sequence}\n',
    }).text.strip()
    if not re.fullmatch(r'ncbiblast-[A-Za-z0-9_-]+', job_id):
        raise RuntimeError(f'Unexpected submission response: {job_id[:100]}')
    logging.info(f'{record.id}: {job_id}', flush=True)
    logging.info('%s: submitted %s', record.id, job_id)

    while True:
        # Sleep even before the first check; do not repeatedly hit the API.
        time.sleep(POLL_INTERVAL)
        status = request(session, 'GET', f'/status/{job_id}').text.strip().upper()
        logging.info(f"Status: {status}")
        if status == 'FINISHED':
            break
        if status in {'ERROR', 'FAILURE', 'NOT_FOUND'}:
            raise ValueError(f'{job_id}: {status}')
        if status not in {'RUNNING', 'PENDING', 'QUEUED'}:
            raise RuntimeError(f'{job_id}: unexpected status {status!r}')

    # Discover the XML result identifier (it may be called xml or outxml).
    types = ET.fromstring(request(session, 'GET', f'/resulttypes/{job_id}').content)
    formats = {node.text for node in types.iter() if node.tag.rsplit('}', 1)[-1] == 'identifier'}
    result_type = next((name for name in ('xml', 'outxml') if name in formats), None)
    if result_type is None:
        raise ValueError(f'{job_id}: no XML result available')
    content = request(session, 'GET', f'/result/{job_id}/{result_type}').content
    ET.fromstring(content)  # Reject malformed downloads before writing the file.
    destination.write_bytes(content)


def main():
    """Read small batches and finish each protein before moving to the next."""
    if EMAIL == 'your@email.com' or '@' not in EMAIL:
        raise ValueError('Set EMAIL at the top of the script to your contact email')
    if BATCH_SIZE < 1 or POLL_INTERVAL < 5 or SUBMISSION_DELAY < 1 or REQUEST_TIMEOUT <= 0:
        raise ValueError('Use a positive batch size/timeout, polling >= 5s and submission delay >= 1s')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=OUTPUT_DIR / 'blast.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    failed = 0
    with INPUT_PATH.open() as source, requests.Session() as session:
        records = SeqIO.parse(source, 'fasta')
        index = 0
        for batch in batched(records, BATCH_SIZE):
            for record in tqdm(batch, desc='BLAST batch', unit='protein'):
                index += 1
                # Include both the input position and ID, so duplicate IDs stay distinct.
                name = re.sub(r'[^A-Za-z0-9_.-]', '_', record.id)[:100]
                destination = OUTPUT_DIR / f'{index:04d}_{name}.xml'
                try:
                    search(session, record, destination)
                except (ValueError, ET.ParseError) as error:
                    logging.error('%s [%s]: %s', destination.name, record.description, error)
                    failed += 1
                except (requests.RequestException, RuntimeError) as error:
                    # The remote job might still be running. Stop rather than submit
                    # another job or retry a POST that may already have succeeded.
                    logging.error('%s [%s]: %s; stopping', destination.name, record.description, error)
                    logging.info('Request failed; see blast.log. Stopping to avoid overlapping jobs.')
                    return 1
                except KeyboardInterrupt:
                    logging.info('%s [%s]: interrupted; remote job may still be running',
                                 destination.name, record.description)
                    return 130
                time.sleep(SUBMISSION_DELAY)
    print(f'Processed {index} proteins; {failed} failed. Results: {OUTPUT_DIR}')
    logging.info(f'Processed {index} proteins; {failed} failed. Results: {OUTPUT_DIR}')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.')
        raise SystemExit(130)
