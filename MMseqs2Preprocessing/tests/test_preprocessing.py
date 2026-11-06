import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'WebViewer'))
import torch
from MMseqs2Preprocessing.preprocess import prepare_dataset
from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import load_prepared
from ModelsAndTrainers.DataLoaders.ProteinClassificationData import ProteinClassificationDataset
from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner
from ModelsAndTrainers.Models.MMseqs2Classifier import MMseqs2Classifier
from ModelsAndTrainers.Trainers.MMseqs2Trainer import MMseqs2Trainer


class FakeRunner:
    version = 'test-version'

    def __init__(self, *args):
        pass

    def clusters(self, records, *args, **kwargs):
        # Each independent cluster has both labels, making the holdout feasible.
        return [(records[i - i % 2]['key'], r['key']) for i, r in enumerate(records)]

    def search(self, queries, reference, root, *args, **kwargs):
        path = Path(root) / 'hits.tsv'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(f"{q['key']}\t{r['key']}\t0.5\t0.8\t0.8\t20\n"
                                for q in queries for r in reference))
        return path


class PreparedWorkflowTest(unittest.TestCase):
    def test_mmseqs_search_flags(self):
        runner = MMseqs2Runner.__new__(MMseqs2Runner)
        runner.version, runner.threads = 'test', 2
        commands = []

        def run(arguments):
            commands.append(arguments)
            if arguments[0] == 'easy-cluster':
                Path(str(arguments[2]) + '_cluster.tsv').write_text('q\tq\n')
            if arguments[0] == 'easy-search':
                Path(arguments[3]).touch()

        runner.run = run
        records = [{'key': 'q', 'sequence': 'MAK', 'target': 0}]
        with tempfile.TemporaryDirectory() as tmp:
            runner.clusters(records, tmp, 0.5, 0.8, 0, 2, 6.0, 0.002, 500)
            cluster = next(c for c in commands if c[0] == 'easy-cluster')
            for flag, value in [('--cluster-mode', 2), ('-s', 6.0), ('-e', 0.002),
                                ('--max-seqs', 500), ('--min-seq-id', 0.5)]:
                self.assertEqual(cluster[cluster.index(flag) + 1], value)
            runner.search(records, records, tmp, 7.5, 0.001, 1000, 0.8, 0)
            search = next(c for c in commands if c[0] == 'easy-search')
            for flag, value in [('-s', 7.5), ('-e', 0.001), ('--max-seqs', 1000),
                                ('-c', 0.8), ('--cov-mode', 0)]:
                self.assertEqual(search[search.index(flag) + 1], value)

    def test_pe_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / 'labels.fasta'
            fasta.write_text(''.join(f'>p{pe} PE={pe}\nMAK\n' for pe in range(1, 6)))
            records = list(ProteinClassificationDataset(fasta))
            self.assertEqual([r['pe'] for r in records], [1, 2, 5])
            self.assertEqual([r['target'] for r in records], [0, 0, 1])
            fasta.write_text('>bad PE=10\nMAK\n')
            with self.assertRaises(ValueError):
                list(ProteinClassificationDataset(fasta))

    def test_prepare_train_resume_without_mmseqs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'input.fasta'
            fasta.write_text(''.join(f'>p{i} PE={1 if i % 2 == 0 else 5}\nM{"A" * (i + 1)}K\n' for i in range(8)))
            output = root / 'dataset' / 'preProcessed.pt'
            with patch('MMseqs2Preprocessing.preprocess.MMseqs2Runner', FakeRunner):
                prepare_dataset(fasta, output, root / 'cache', report=lambda _: None)
            state, digest = load_prepared(output.parent)
            self.assertEqual(state['train']['features'].shape[1], 11)
            self.assertEqual(set(state['validation']['targets'].tolist()), {0, 1})
            # Direct file paths and the dataset-directory path used by WebViewer agree.
            self.assertEqual(load_prepared(output)[1], digest)
            extra = output.parent / 'other.pt'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'Multiple .pt files'):
                load_prepared(output.parent)
            extra.unlink()
            config = {'dataset_paths': {'training_data': str(output.parent)},
                      'hyperparameters': {'epochs': 1, 'batch_size': 2}}
            with patch.dict(os.environ, {'PATH': ''}), patch('subprocess.Popen', side_effect=AssertionError('Training launched a subprocess')):
                trainer = MMseqs2Trainer(config, root / 'run', lambda _: None)
                trainer.bind_models({'classifier': MMseqs2Classifier()})
                trainer.train()
                self.assertEqual(trainer.get_status(), 'COMPLETED')
                checkpoint = root / 'run/checkpoints/checkpoint_0001.pt'
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved['reference'], state['reference'])
                self.assertEqual(saved['prepared_digest'], digest)
                with patch('ModelsAndTrainers.Models.MMseqs2Classifier.MMseqs2Runner', FakeRunner):
                    result = MMseqs2Classifier().infer({'sequence': 'MACK'}, checkpoint)
                self.assertEqual([c['label'] for c in result['classes']], ['PE 1–2', 'PE 5'])
                self.assertAlmostEqual(sum(c['probability'] for c in result['classes']), 1, places=6)
                resumed = MMseqs2Trainer(config, root / 'resumed', lambda _: None)
                resumed.bind_models({'classifier': MMseqs2Classifier()})
                resumed.load_checkpoint(checkpoint)
                resumed.train()
                state['train']['features'][0, 0] += 0.1
                torch.save(state, output)
                with self.assertRaisesRegex(ValueError, 'changed since checkpoint'):
                    resumed.train()
            canonical = output.parent / 'prepared.pt'
            torch.save(state, canonical)
            self.assertEqual(load_prepared(output.parent)[1], load_prepared(canonical)[1])
            state['feature_version'] = -1
            torch.save(state, output)
            with self.assertRaisesRegex(ValueError, 'version'):
                load_prepared(output)


if __name__ == '__main__':
    unittest.main()
