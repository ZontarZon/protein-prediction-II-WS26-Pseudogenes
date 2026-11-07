import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'WebViewer'))
import torch
from MMseqs2Preprocessing.preprocess import prepare_dataset
from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import load_prepared
from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner
from ModelsAndTrainers.Models.ProteinPEClassifier import ProteinPEClassifier
from ModelsAndTrainers.Trainers.MMseqs2Trainer import MMseqs2Trainer
from ModelsAndTrainers.Models.ProteinPEClassifier import encoder_cache_directory
from backend.model_registry import ModelRegistry
from backend.trainer_registry import TrainerRegistry
from backend.dataset_preview import loader_preview
from backend.dataset_statistics import DatasetInventory


class FakeRunner:
    version = 'test-version'

    def __init__(self, *args):
        pass

    def clusters(self, records, *args, **kwargs):
        # Each independent cluster has both labels, making the holdout feasible.
        return [(records[i - i % 2]['key'], r['key']) for i, r in enumerate(records)]

    def search(self, queries, reference, root, *args, **kwargs):
        raise AssertionError('Sequence-split preprocessing must not run a feature search')



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

    def test_frozen_encoder_and_sequence_pooling(self):
        model = ProteinPEClassifier(hidden_size=4)

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4, 1024)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        def tokenize(sequences, **kwargs):
            length = len(sequences[0].split()) + 1
            return {'input_ids': torch.ones((1, length), dtype=torch.long),
                    'attention_mask': torch.ones((1, length), dtype=torch.long)}

        model.encoder = Encoder().requires_grad_(False)
        model.tokenizer = tokenize
        model.train()
        self.assertFalse(model.encoder.training)
        encoded = model.embed(['MAK', 'M' * 1025])
        self.assertEqual(tuple(encoded.shape), (2, 1024))
        self.assertFalse(encoded.requires_grad)
        self.assertTrue(torch.allclose(encoded[0], model.encoder.embedding.weight[1]))
        logits = model(['MAK', 'MKK'])
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.head[0].weight.grad)
        self.assertIsNone(model.encoder.embedding.weight.grad)

    def test_registry_model_selection(self):
        project = Path(__file__).resolve().parents[2] / 'WebViewer'
        training = project / 'ModelsAndTrainers'
        trainers = TrainerRegistry(training, project)
        models = ModelRegistry(training, project)
        declaration = trainers.get('MMseqs2Trainer').configuration_schema()['models'][0]
        # Match the filtering performed by NewRun.tsx.
        compatible = [item for item in models.catalog()
                      if item['id'] in declaration['choices']
                      and declaration['interface'] in item['interfaces']]
        self.assertEqual([item['id'] for item in compatible], ['ProteinPEClassifier'])
        trainer = trainers.get('MMseqs2Trainer')({}, Path('/tmp/unused-mmseqs-run'), lambda _: None)
        trainer.bind_models({'classifier': models.create('ProteinPEClassifier')})
        self.assertEqual(trainer.net.model_id, 'ProteinPEClassifier')

    def test_encoder_cache_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'HF_HOME': tmp}, clear=True):
                self.assertEqual(encoder_cache_directory(), str(Path(tmp) / 'hub'))
            explicit = Path(tmp) / 'explicit'
            with patch.dict(os.environ, {'HF_HOME': tmp, 'HF_HUB_CACHE': str(explicit)}, clear=True):
                self.assertEqual(encoder_cache_directory(), str(explicit))
            self.assertTrue(explicit.is_dir())
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(Path, 'mkdir'):
                    cache = encoder_cache_directory()
                self.assertEqual(Path(cache), Path(__file__).resolve().parents[2] / 'WebViewer/.cache/huggingface/hub')

    def test_prepare_train_resume_without_mmseqs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fasta = root / 'input.fasta'
            fasta.write_text(''.join(f'>p{i} PE={1 if i % 2 == 0 else 5}\nM{"A" * (i + 1)}K\n' for i in range(8)))
            annotations = root / 'pseudogene_caution.tsv'
            annotations.write_text('Entry\tCaution\n' + ''.join(
                f'p{i}\tCould be the product of a pseudogene\n' for i in range(1, 8, 2)))
            output = root / 'dataset' / 'preProcessed.pt'
            with patch('MMseqs2Preprocessing.preprocess.MMseqs2Runner', FakeRunner):
                prepare_dataset(fasta, output, root / 'cache', report=lambda _: None,
                                annotation_path=annotations)
            state, digest = load_prepared(output.parent)
            self.assertNotIn('features', state['train'])
            self.assertTrue(all('sequence' in r for r in state['validation']['records']))
            self.assertEqual(set(state['validation']['targets'].tolist()), {0, 1})
            # Direct file paths and the dataset-directory path used by WebViewer agree.
            self.assertEqual(load_prepared(output)[1], digest)
            extra = output.parent / 'other.pt'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'Multiple .pt files'):
                load_prepared(output.parent)
            extra.unlink()
            folder = Path(__file__).resolve().parents[2] / 'WebViewer/ModelsAndTrainers/DataLoaders'
            preview, _ = loader_preview(output, folder)
            self.assertEqual(preview['loader'], 'MMseqs2ClassificationData')
            self.assertIn('sequence', preview['view']['columns'])
            self.assertEqual(len(preview['view']['rows']), 8)
            filtered, _ = loader_preview(output, folder, search={
                'filters': [{'column': 'split', 'operation': 'equals', 'value': 'validation'}]})
            self.assertEqual(len(filtered['view']['rows']), len(state['validation']['targets']))
            inventory = DatasetInventory.dataset_statistics(output.parent, 'inventory')
            self.assertEqual(inventory['total'], 8)
            self.assertEqual(sum(g['count'] for g in inventory['groups']), 8)
            config = {'dataset_paths': {'training_data': str(output.parent)},
                      'hyperparameters': {'epochs': 1, 'batch_size': 2}}
            observed = []

            def embed(model, sequences):
                self.assertTrue(all(isinstance(sequence, str) for sequence in sequences))
                observed.extend(sequences)
                return torch.tensor([[len(s) / 100.] * 1024 for s in sequences],
                                    device=next(model.head.parameters()).device)

            with patch.dict(os.environ, {'PATH': ''}), patch('subprocess.Popen', side_effect=AssertionError('Training launched a subprocess')), patch.object(ProteinPEClassifier, 'setup_encoder'), patch.object(ProteinPEClassifier, 'embed', embed):
                trainer = MMseqs2Trainer(config, root / 'run', lambda _: None)
                trainer.bind_models({'classifier': ProteinPEClassifier()})
                initial = trainer.net.head[0].weight.detach().clone()
                trainer.train()
                self.assertFalse(torch.equal(initial, trainer.net.head[0].weight))
                self.assertEqual(set(observed), {r['sequence'] for split in ('train', 'validation') for r in state[split]['records']})
                self.assertEqual(trainer.get_status(), 'COMPLETED')
                checkpoint = root / 'run/checkpoints/checkpoint_0001.pt'
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved['encoder_checkpoint'], 'Rostlab/prot_t5_xl_half_uniref50-enc')
                self.assertTrue(all(k.startswith('head.') for k in saved['model_state_dict']))
                self.assertEqual(saved['prepared_digest'], digest)
                result = ProteinPEClassifier().infer({'sequence': 'MACK'}, checkpoint)
                self.assertEqual(len(result['classes']), 2)
                self.assertAlmostEqual(sum(c['probability'] for c in result['classes']), 1, places=6)
                resumed = MMseqs2Trainer(config, root / 'resumed', lambda _: None)
                resumed.bind_models({'classifier': ProteinPEClassifier()})
                resumed.load_checkpoint(checkpoint)
                resumed.epochs = 2
                resumed.train()
                self.assertEqual(resumed.start_epoch, 2)
                self.assertGreater(resumed.step, trainer.step)
                state['train']['records'][0]['sequence'] += 'A'
                torch.save(state, output)
                with self.assertRaisesRegex(ValueError, 'changed since checkpoint'):
                    resumed.train()
            canonical = output.parent / 'prepared.pt'
            torch.save(state, canonical)
            self.assertEqual(load_prepared(output.parent)[1], load_prepared(canonical)[1])
            state['prepared_version'] = 2
            torch.save(state, output)
            with self.assertRaisesRegex(ValueError, 'Regenerate'):
                load_prepared(output)


if __name__ == '__main__':
    unittest.main()
