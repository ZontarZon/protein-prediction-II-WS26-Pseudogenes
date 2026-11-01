"""CPU MMseqs2 commands with cooperative control and content-addressed caches."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


class MMseqs2Stopped(Exception):
    """The owning training run requested cancellation."""


class MMseqs2Runner:
    def __init__(self, executable='mmseqs', threads=1, control=None, paused=None):
        self.executable = shutil.which(str(executable))
        if self.executable is None:
            raise ValueError('MMseqs2 executable not found. Build external/MMseqs2 or install mmseqs and configure its executable path.')
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError('MMseqs2 threads must be positive')
        self.control = control or (lambda: True)
        self.paused = paused or (lambda: False)
        self.version = self.run(['version']).strip()

    def run(self, arguments):
        if not self.control():
            raise MMseqs2Stopped()
        # Files prevent pipe deadlocks and keep large command logs out of RAM.
        import tempfile
        with tempfile.TemporaryFile(mode='w+t') as output:
            process = subprocess.Popen([self.executable, *map(str, arguments)], stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    suspended = self.paused()
                    if suspended:
                        self._signal(process.pid, signal.SIGSTOP)
                    if not self.control():
                        raise MMseqs2Stopped()
                    if suspended:
                        self._signal(process.pid, signal.SIGCONT)
                    time.sleep(.1)
                if process.returncode:
                    output.seek(0, os.SEEK_END)
                    size = output.tell()
                    output.seek(max(0, size - 4000))
                    raise RuntimeError(f'MMseqs2 {arguments[0]} failed ({process.returncode}): {output.read()}')
                output.seek(0)
                return output.read() if arguments[0] == 'version' else ''
            finally:
                if process.poll() is None:
                    self._signal(process.pid, signal.SIGCONT)
                    self._signal(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._signal(process.pid, signal.SIGKILL)
                        process.wait()

    @staticmethod
    def _signal(pid, value):
        try:
            os.killpg(pid, value)
        except ProcessLookupError:
            pass

    @staticmethod
    def write_fasta(records, path):
        with Path(path).open('w') as stream:
            for record in records:
                stream.write(f">{record['key']}\n{record['sequence']}\n")

    def prepare(self, records, root, settings, operation):
        signature = hashlib.sha256(json.dumps([self.version, records, settings, operation], sort_keys=True).encode()).hexdigest()
        work = Path(root) / signature
        if work.exists() and not (work / 'complete.json').exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        return work

    def clusters(self, records, root, identity, coverage):
        settings = {'identity': identity, 'coverage': coverage}
        work = self.prepare(records, root, settings, 'cluster')
        result = work / 'clusters_cluster.tsv'
        if not (work / 'complete.json').exists():
            self.write_fasta(records, work / 'input.fasta')
            self.run(['easy-cluster', work / 'input.fasta', work / 'clusters', work / 'tmp',
                      '--min-seq-id', identity, '-c', coverage, '--cov-mode', '0', '--dbtype', '1', '--threads', self.threads])
            if not result.is_file():
                raise RuntimeError('MMseqs2 did not produce cluster assignments')
            (work / 'complete.json').write_text(json.dumps(settings))
        return [line.split('\t') for line in result.read_text().splitlines() if line]

    def search(self, queries, reference, root, sensitivity, evalue):
        settings = {'sensitivity': sensitivity, 'evalue': evalue}
        work = self.prepare([queries, reference], root, settings, 'search')
        result = work / 'hits.tsv'
        if not (work / 'complete.json').exists():
            self.write_fasta(queries, work / 'query.fasta')
            database = self.prepare(reference, root, {'dbtype': 1}, 'reference_database')
            if not (database / 'complete.json').exists():
                self.write_fasta(reference, database / 'reference.fasta')
                self.run(['createdb', database / 'reference.fasta', database / 'reference_db', '--dbtype', '1'])
                (database / 'complete.json').write_text('{}')
            self.run(['easy-search', work / 'query.fasta', database / 'reference_db', result, work / 'tmp',
                      '-s', sensitivity, '-e', evalue, '--threads', self.threads, '--search-type', '1', '--dbtype', '1',
                      '--alignment-mode', '3', '--max-seqs', '1000', '--format-output', 'query,target,fident,qcov,tcov,bits'])
            if not result.is_file():
                raise RuntimeError('MMseqs2 did not produce search results')
            (work / 'complete.json').write_text(json.dumps(settings))
        return result
