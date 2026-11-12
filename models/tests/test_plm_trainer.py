"""Exercise training without downloading the multi-gigabyte ProtT5 encoder."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import (
    DEFAULT_SETTINGS, PREPARED_VERSION,
)
from models.ModelsAndTrainers.Models.ProteinPEClassifier import ProteinPEClassifier
from models.ModelsAndTrainers.Trainers.PLM_MLPTrainer import PLM_MLPTrainer


def fake_embeddings(model, sequences):
    device = next(model.head.parameters()).device
    return torch.tensor([[len(sequence) / 10] * 1024 for sequence in sequences], device=device)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        state = {
            'prepared_version': PREPARED_VERSION,
            'dataset_kind': 'mmseqs2_sequence_split',
            'settings': DEFAULT_SETTINGS.copy(),
            'mmseqs_version': 'test', 'dataset_digest': 'test-dataset',
        }
        for name, sequences in [('train', ['AAA', 'CCCC', 'DDDDD', 'EEEEEE']),
                                ('validation', ['FFFFFFF', 'GGGGGGGG'])]:
            state[name] = {
                'records': [{'key': f'{name}{i}', 'sequence': sequence, 'target': i % 2}
                            for i, sequence in enumerate(sequences)],
                'targets': torch.tensor([i % 2 for i in range(len(sequences))]),
            }
        self.data = self.root / 'prepared.pt'
        torch.save(state, self.data)
        self.config = {
            'training_data': str(self.data), 'run_dir': str(self.root / 'run'),
            'device': 'cpu', 'epochs': 2, 'batch_size': 2,
            'hidden_sizes': [8], 'dropout': 0.2,
        }
        self.encoder = patch.object(ProteinPEClassifier, 'setup_encoder')
        self.embedding = patch.object(ProteinPEClassifier, 'embed', fake_embeddings)
        self.encoder.start()
        self.embedding.start()
        self.addCleanup(self.encoder.stop)
        self.addCleanup(self.embedding.stop)

    def test_training_updates_weights_and_resume_matches_uninterrupted_run(self):
        trainer = PLM_MLPTrainer(self.config)
        initial = {key: value.clone() for key, value in trainer.net.head.state_dict().items()}
        trainer.train()
        self.assertEqual(trainer.step, 4)
        self.assertTrue(any(not torch.equal(initial[key], value)
                            for key, value in trainer.net.head.state_dict().items()))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in trainer.metrics.values()))
        checkpoint = self.root / 'run/checkpoints/checkpoint_0001.pt'
        resumed = PLM_MLPTrainer(self.config)
        resumed.load_checkpoint(checkpoint)
        resumed.train()
        self.assertEqual(resumed.metrics, trainer.metrics)
        for key, value in trainer.net.head.state_dict().items():
            torch.testing.assert_close(resumed.net.head.state_dict()[key], value, rtol=0, atol=0)

    def test_resume_rejects_changed_dataset(self):
        trainer = PLM_MLPTrainer({**self.config, 'epochs': 1})
        trainer.train()
        resumed = PLM_MLPTrainer(self.config)
        resumed.load_checkpoint(self.root / 'run/checkpoints/checkpoint_0001.pt')
        state = torch.load(self.data, weights_only=True)
        state['mmseqs_version'] = 'changed'
        torch.save(state, self.data)
        with self.assertRaisesRegex(ValueError, 'Preprocessed dataset changed'):
            resumed.train()

    def test_invalid_hyperparameters(self):
        for invalid in [{'epochs': 0}, {'batch_size': True}, {'learning_rate': float('nan')},
                        {'weight_decay': -1}, {'seed': -1}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PLM_MLPTrainer({**self.config, **invalid})


if __name__ == '__main__':
    unittest.main()
