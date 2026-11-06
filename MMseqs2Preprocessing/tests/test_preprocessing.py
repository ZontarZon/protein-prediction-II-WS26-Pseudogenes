import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'WebViewer'))
import torch
from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import prepare_dataset, load_prepared
from ModelsAndTrainers.Models.MMseqs2Classifier import MMseqs2Classifier
from ModelsAndTrainers.Trainers.MMseqs2Trainer import MMseqs2Trainer


class FakeRunner:
    version = 'test-version'

    def __init__(self, *args):
        pass

    def clusters(self, records, *args):
        # Each independent cluster has both labels, making the holdout feasible.
        return [(records[i - i % 2]['key'], r['key']) for i, r in enumerate(records)]

    def search(self, queries, reference, root, *args):
        path = Path(root) / 'hits.tsv'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(f"{q['key']}\t{r['key']}\t0.5\t0.8\t0.8\t20\n"
                                for q in queries for r in reference))
        return path


class PreparedWorkflowTest(unittest.TestCase):
    def test_prepare_train_resume_without_mmseqs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'input.fasta'
            fasta.write_text(''.join(f'>p{i} PE={1 if i % 2 == 0 else 5}\nM{"A" * (i + 1)}K\n' for i in range(8)))
            output = root / 'dataset' / 'prepared.pt'
            with patch('ModelsAndTrainers.DataLoaders.MMseqs2Runner.MMseqs2Runner', FakeRunner):
                prepare_dataset(fasta, output, root / 'cache', report=lambda _: None)
            state, digest = load_prepared(output.parent)
            self.assertEqual(state['train']['features'].shape[1], 11)
            self.assertEqual(set(state['validation']['targets'].tolist()), {0, 1})
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
                resumed = MMseqs2Trainer(config, root / 'resumed', lambda _: None)
                resumed.bind_models({'classifier': MMseqs2Classifier()})
                resumed.load_checkpoint(checkpoint)
                resumed.train()
                state['train']['features'][0, 0] += 0.1
                torch.save(state, output)
                with self.assertRaisesRegex(ValueError, 'changed since checkpoint'):
                    resumed.train()
            state['feature_version'] = -1
            torch.save(state, output)
            with self.assertRaisesRegex(ValueError, 'version'):
                load_prepared(output)


if __name__ == '__main__':
    unittest.main()
