"""Full-dataset PE classification with disk caching and a sequence-based holdout."""
import hashlib
import json
import random
import secrets
import sqlite3
import math
import time

import torch
from torch import nn

from backend.trainers.base import TrainerBase, dataset, hyperparameter, model
from ModelsAndTrainers.DataLoaders.ProteinClassificationData import ProteinClassificationDataset
from ModelsAndTrainers.Models.ProteinPEClassifier import sequence_input


class ProteinPETrainer(TrainerBase):
    classifier = model('Protein classifier', choices=['ProteinPEClassifier'], interface='protein_pe_classifier')
    training_data = dataset('PE-labelled FASTA data', 'A FASTA file or directory of plain/gzip FASTA files. PE 1–4 = true; PE 5 = false.')
    epochs = hyperparameter('Epochs', 'integer', 3)
    batch_size = hyperparameter('Batch size', 'integer', 2, minimum=1)
    learning_rate = hyperparameter('Learning rate', 'float', 0.001)
    subset_real_count = hyperparameter('Subset count · Real', 'integer', 0, description='0 uses all PE 1–4 entries. A positive count samples at most this many per run.', dataset_control=True, minimum=0)
    subset_fake_count = hyperparameter('Subset count · Fake', 'integer', 0, description='0 uses all PE 5 entries. A positive count samples at most this many per run.', dataset_control=True, minimum=0)
    validation_percent = hyperparameter('Validation percentage', 'integer', 20, minimum=1, maximum=50)

    def __init__(self, config, run_dir, report):
        super().__init__(config, run_dir, report)
        self.device = torch.device(config.get('device', 'cpu'))
        torch.manual_seed(int(config.get('seed', 7)))
        for key in ('epochs', 'batch_size'):
            if int(getattr(self, key)) < 1:
                raise ValueError(f'{key} must be positive')
        if not 1 <= int(self.validation_percent) <= 50:
            raise ValueError('validation_percent must be between 1 and 50')
        if not 0 < float(self.learning_rate) < float('inf'):
            raise ValueError('learning_rate must be finite and positive')
        for key in ('subset_real_count', 'subset_fake_count'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        self.selection_seed = secrets.randbits(63)
        self.dataset_digest = None
        self.net = None
        self.optimizer = None
        self.start_epoch = 0
        self.step = 0

    def bind_models(self, models):
        super().bind_models(models)
        self.net = models.get('classifier')
        # Registries import source files under dynamic module names.
        if getattr(self.net, 'model_id', None) != 'ProteinPEClassifier':
            raise ValueError('Select ProteinPEClassifier for the classifier role')
        if not any(p.requires_grad for p in self.net.head.parameters()):
            raise ValueError('The classification head must be trainable; disable freezing for this model role')
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.head.parameters(), lr=float(self.learning_rate))

    @classmethod
    def dataset_statistics(cls, path, role):
        if role != 'training_data':
            return None
        counts = [0, 0]
        for record in ProteinClassificationDataset(path):
            sequence_input(record['sequence'])  # Use the same validity rules as training.
            counts[record['target']] += 1
        return {'objective': 'classification', 'total': sum(counts),
                'groups': [{'key': 'real', 'label': 'Class 1 · Real (PE 1–4)', 'count': counts[0], 'subset_parameter': 'subset_real_count'},
                           {'key': 'fake', 'label': 'Class 2 · Fake (PE 5)', 'count': counts[1], 'subset_parameter': 'subset_fake_count'}],
                'validation_parameter': 'validation_percent',
                'note': 'Batch counts are estimates: identical sequences stay together, which may slightly change the split.'}

    def _prepare_data(self, connection):
        """Index all entries, select a per-run subset, then split that subset."""
        connection.execute('DROP TABLE IF EXISTS samples')
        connection.execute('CREATE TABLE samples (id INTEGER PRIMARY KEY, sequence TEXT, target INTEGER, validation INTEGER DEFAULT 0, selected INTEGER DEFAULT 0, digest TEXT)')
        digest = hashlib.sha256()
        for index, record in enumerate(ProteinClassificationDataset(self.dataset_path('training_data')), 1):
            if not self.check_control():
                return [[0, 0], [0, 0]]
            sequence = sequence_input(record['sequence'])
            target = record['target']
            sequence_digest = hashlib.sha256(sequence.encode()).hexdigest()
            digest.update(f'{target}:{sequence_digest}\n'.encode())
            connection.execute('INSERT INTO samples(sequence,target,digest) VALUES (?,?,?)', (sequence, target, sequence_digest))
            if index % 1000 == 0:
                self.report({'phase': 'indexing', 'indexed_records': index, 'step': self.step,
                             'epoch': self.start_epoch + 1, 'total_epochs': int(self.epochs),
                             'progress': 0., 'validation_progress': 0.})
        fingerprint = digest.hexdigest()
        if self.dataset_digest is not None and self.dataset_digest != fingerprint:
            raise ValueError('Dataset entries/order changed since the checkpoint; refusing to change its subset on resume')
        self.dataset_digest = fingerprint
        connection.execute('CREATE INDEX samples_digest ON samples(digest)')
        rng = random.Random(self.selection_seed)
        totals = []
        for target, limit in enumerate((self.subset_real_count, self.subset_fake_count)):
            available = [row[0] for row in connection.execute('SELECT id FROM samples WHERE target=? ORDER BY id', (target,))]
            chosen = rng.sample(available, min(limit, len(available))) if limit else available
            totals.append(len(chosen))
            connection.executemany('UPDATE samples SET selected=1 WHERE id=?', ((key,) for key in chosen))
        group_counts = {}
        for sequence_digest, target, count in connection.execute('SELECT digest,target,COUNT(*) FROM samples WHERE selected=1 GROUP BY digest,target ORDER BY digest,target'):
            group_counts.setdefault(sequence_digest, [0, 0])[target] = count
        assigned = set()
        validation_counts = [0, 0]
        for target in (0, 1):
            groups = [key for key, counts in group_counts.items() if counts[target]]
            rng.shuffle(groups)
            desired = max(1, min(totals[target] - 1, int(totals[target] * int(self.validation_percent) / 100 + .5))) if totals[target] >= 2 else 0
            for sequence_digest in groups:
                if validation_counts[target] >= desired:
                    break
                if sequence_digest in assigned:
                    continue
                counts = group_counts[sequence_digest]
                # A shared sequence can have different evidence labels across
                # entries. Keep all its selected copies together, across classes,
                # while leaving at least one entry of each class for training.
                if any(count and validation_counts[t] + count >= totals[t] for t, count in enumerate(counts)):
                    continue
                connection.execute('UPDATE samples SET validation=1 WHERE selected=1 AND digest=?', (sequence_digest,))
                assigned.add(sequence_digest)
                validation_counts = [current + count for current, count in zip(validation_counts, counts)]
        counts = [[0, 0], [0, 0]]
        for split, target, count in connection.execute('SELECT validation,target,COUNT(*) FROM samples WHERE selected=1 GROUP BY validation,target'):
            counts[split][target] = count
        connection.commit()
        return counts

    def train(self):
        self._status = 'RUNNING'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        self.report({'phase': 'indexing', 'step': self.step, 'epoch': self.start_epoch + 1,
                     'total_epochs': int(self.epochs), 'progress': 0., 'validation_progress': 0.})
        with sqlite3.connect(self.run_dir / 'protein_samples.sqlite') as connection:
            coverage = self._prepare_data(connection)
            if self.stop_requested:
                self._status = 'STOPPED'
                return
            if any(count == 0 for split in coverage for count in split):
                raise ValueError(f'Both PE classes are required in each split; observed train/validation counts {coverage}. Select at least two distinct sequences per class, increase subset counts (0 = all), or adjust validation_percent.')
            ids = [[row[0] for row in connection.execute('SELECT id FROM samples WHERE selected=1 AND validation=? ORDER BY id', (split,))]
                   for split in (0, 1)]
            self.report({**self.get_metrics(), 'phase': 'loading_encoder'})
            self.net.setup_encoder()
            train_steps = math.ceil(len(ids[0]) / int(self.batch_size))
            validation_steps = math.ceil(len(ids[1]) / int(self.batch_size))
            for epoch in range(self.start_epoch, int(self.epochs)):
                rng = random.Random(self.selection_seed + epoch)
                # Every record in each partition participates in a full permutation.
                for partition in ids:
                    partition.sort()
                    rng.shuffle(partition)
                summary = {}
                for split, phase in enumerate(('training', 'validation')):
                    loss_total, correct, seen = 0., 0, 0
                    self.net.train(split == 0)
                    self.report({'phase': phase, 'epoch': epoch + 1, 'total_epochs': int(self.epochs),
                                 'step': self.step, 'epoch_step': 0 if split == 0 else train_steps,
                                 'epoch_steps': train_steps, 'validation_step': 0,
                                 'validation_steps': validation_steps, 'progress': 0. if split == 0 else 100.,
                                 'validation_progress': 0., **summary})
                    for offset in range(0, len(ids[split]), int(self.batch_size)):
                        if not self.check_control():
                            break
                        batch = [connection.execute('SELECT sequence,target FROM samples WHERE id=?', (key,)).fetchone()
                                 for key in ids[split][offset:offset + int(self.batch_size)]]
                        labels = torch.tensor([item[1] for item in batch], device=self.device)
                        with torch.set_grad_enabled(split == 0):
                            logits = self.net([item[0] for item in batch])
                            loss = nn.functional.cross_entropy(logits, labels)
                        if split == 0:
                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            self.optimizer.step()
                            self.step += 1
                        loss_total += loss.item() * len(batch)
                        correct += int((logits.argmax(-1) == labels).sum().item())
                        seen += len(batch)
                        batch_number = offset // int(self.batch_size) + 1
                        elapsed = time.monotonic() - started
                        self.report({'phase': phase, 'epoch': epoch + 1, 'total_epochs': int(self.epochs),
                                     'step': self.step, 'epoch_position': epoch + seen / len(ids[split]), 'epoch_step': batch_number if split == 0 else train_steps,
                                     'epoch_steps': train_steps, 'validation_step': batch_number if split else 0,
                                     'validation_steps': validation_steps,
                                     'progress': 100. * seen / len(ids[0]) if split == 0 else 100.,
                                     'validation_progress': 100. * seen / len(ids[1]) if split else 0.,
                                     'loss' if split == 0 else 'validation_loss': loss_total / seen,
                                     'accuracy' if split == 0 else 'validation_accuracy': correct / seen,
                                     'learning_rate': self.optimizer.param_groups[0]['lr'],
                                     'elapsed_seconds': elapsed,
                                     **(summary if split else {})})
                    if self.stop_requested:
                        break
                    summary.update({'loss' if split == 0 else 'validation_loss': loss_total / seen,
                                    'accuracy' if split == 0 else 'validation_accuracy': correct / seen})
                if self.stop_requested:
                    break
                self.start_epoch = epoch + 1
                self.report({**self.get_metrics(), **summary, 'phase': 'epoch_complete',
                             'validation_progress': 0., 'validation_step': 0})
                self.save_checkpoint(self.run_dir / 'checkpoints' / f'checkpoint_{self.start_epoch:04d}.pt')
        self._status = 'STOPPED' if self.stop_requested else 'COMPLETED'

    def save_checkpoint(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Encoder weights remain the immutable HF checkpoint; store only the trained head.
        state = {'model_state_dict': {f'head.{k}': v for k, v in self.net.head.state_dict().items()},
                 'model_settings': {'hidden_size': self.net.hidden_size},
                 'encoder_checkpoint': 'Rostlab/prot_t5_xl_half_uniref50-enc',
                 'optimizer': self.optimizer.state_dict(), 'epoch': self.start_epoch, 'step': self.step,
                 'config': self.config, 'metrics': self.get_metrics(), 'rng_state': torch.get_rng_state(),
                 'selection_seed': self.selection_seed, 'dataset_digest': self.dataset_digest,
                 'subset_counts': [self.subset_real_count, self.subset_fake_count],
                 'validation_percent': int(self.validation_percent)}
        torch.save(state, path)
        path.with_suffix('.json').write_text(json.dumps({'epoch': self.start_epoch, 'step': self.step, 'models': self.config.get('models', {}), **self.get_metrics()}, indent=2))

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        expected = {'hidden_size': self.net.hidden_size}
        if state['model_settings'].get('hidden_size') != expected['hidden_size']:
            raise ValueError('Checkpoint head architecture differs from this run')
        if state.get('subset_counts', [0, 0]) != [self.subset_real_count, self.subset_fake_count] or state.get('validation_percent', int(self.validation_percent)) != int(self.validation_percent):
            raise ValueError('Keep the original subset counts and validation percentage when resuming')
        if 'selection_seed' not in state:
            raise ValueError('This checkpoint predates fixed per-run subsets; start a new run')
        self.selection_seed = state['selection_seed']
        self.dataset_digest = state.get('dataset_digest')
        self.net.load_weight_state(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.start_epoch, self.step = state['epoch'], state['step']
        self._metrics.update(state.get('metrics', {}))
        torch.set_rng_state(state['rng_state'].cpu())
