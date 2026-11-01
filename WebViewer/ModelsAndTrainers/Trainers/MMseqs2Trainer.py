"""Independent CPU training workflow using MMseqs2-derived PE features."""
import json
import math

import torch
from torch import nn

from backend.trainers.base import TrainerBase, dataset, hyperparameter, model
from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner, MMseqs2Stopped
from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import (
    FEATURE_VERSION, DEFAULT_SETTINGS, load_records, fingerprint, split_records, features,
)


class MMseqs2Trainer(TrainerBase):
    classifier = model('MMseqs2 classifier', choices=['MMseqs2Classifier'], interface='mmseqs2_pe_classifier')
    training_data = dataset('PE-labelled protein FASTA', 'Plain/gzip FASTA: PE=1..4 maps to class 0, PE=5 to class 1.')
    executable = hyperparameter('MMseqs2 executable', 'text', 'mmseqs')
    threads = hyperparameter('MMseqs2 CPU threads', 'integer', 2, minimum=1)
    epochs = hyperparameter('Epochs', 'integer', 10, minimum=1)
    batch_size = hyperparameter('Batch size', 'integer', 64, minimum=1)
    learning_rate = hyperparameter('Learning rate', 'float', .001, minimum=.000001)
    identity = hyperparameter('Cluster minimum identity', 'float', .5, minimum=0., maximum=1.)
    coverage = hyperparameter('Cluster coverage', 'float', .8, minimum=0., maximum=1.)
    sensitivity = hyperparameter('Search sensitivity', 'float', 5.7, minimum=1., maximum=7.5)
    evalue = hyperparameter('Search E-value threshold', 'float', .001, minimum=0.)
    validation_percent = hyperparameter('Validation percentage', 'integer', 20, minimum=1, maximum=50)
    seed = hyperparameter('Cluster split seed', 'integer', 7, minimum=0)

    def __init__(self, config, run_dir, report):
        super().__init__(config, run_dir, report)
        for key in ('threads', 'epochs', 'batch_size', 'validation_percent', 'seed'):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == 'seed' else 1):
                raise ValueError(f'{key} must be a valid integer')
        for key in ('learning_rate', 'identity', 'coverage', 'sensitivity', 'evalue'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        if not 1 <= self.validation_percent <= 50 or not 0 <= self.identity <= 1 or not 0 < self.coverage <= 1:
            raise ValueError('Invalid cluster/split thresholds')
        if not 1 <= self.sensitivity <= 7.5 or self.evalue <= 0 or self.learning_rate <= 0:
            raise ValueError('Invalid search or optimizer settings')
        self.settings = {key: getattr(self, key) for key in DEFAULT_SETTINGS}
        self.start_epoch = self.step = 0
        self.dataset_digest = None
        self.reference = []
        self.mmseqs_version = None
        self.optimizer = self.net = None

    def bind_models(self, models):
        super().bind_models(models)
        self.net = models.get('classifier')
        if getattr(self.net, 'model_id', None) != 'MMseqs2Classifier':
            raise ValueError('Select MMseqs2Classifier')
        if not any(p.requires_grad for p in self.net.parameters()):
            raise ValueError('Classifier weights must be trainable')
        self.net.to('cpu')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def get_device_info(self):
        return {'type': 'cpu', 'id': 'cpu', 'name': 'CPU (MMseqs2 and classifier)'}

    def get_performance_info(self):
        return {'mmseqs_threads': self.threads, 'device': 'cpu'}

    def train(self):
        self._status = 'RUNNING'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._train()
        except MMseqs2Stopped:
            self._status = 'STOPPED'
            return
        self._status = 'STOPPED' if self.stop_requested else 'COMPLETED'

    def _train(self):
        if not self.check_control():
            return
        self.report({'phase': 'loading_fasta', 'step': self.step})
        records = load_records(self.dataset_path('training_data'), self.check_control)
        digest = fingerprint(records)
        if self.dataset_digest is not None and self.dataset_digest != digest:
            raise ValueError('Dataset changed since checkpoint')
        self.dataset_digest = digest
        runner = MMseqs2Runner(self.executable, self.threads, self.check_control, lambda: self.paused)
        if self.mmseqs_version is not None and self.mmseqs_version != runner.version:
            raise ValueError('Resume requires the same MMseqs2 version')
        self.mmseqs_version = runner.version
        cache = self.run_dir / 'mmseqs2_cache'
        self.report({'phase': 'clustering'})
        assignments = runner.clusters(records, cache, self.identity, self.coverage)
        train, valid = split_records(records, assignments, self.settings)
        self.reference = train
        (self.run_dir / 'mmseqs2_manifest.json').write_text(json.dumps({
            'dataset_digest': digest, 'mmseqs_version': runner.version, 'settings': self.settings,
            'feature_version': FEATURE_VERSION, 'train': [r['key'] for r in train],
            'validation': [r['key'] for r in valid],
            'records': [{'key': r['key'], 'original_id': r['id']} for r in records],
        }, indent=2))
        prepared = []
        for phase, partition in (('training_features', train), ('validation_features', valid)):
            if not self.check_control():
                return
            self.report({'phase': phase})
            hits = runner.search(partition, train, cache, self.sensitivity, self.evalue)
            prepared.append((features(partition, train, hits), torch.tensor([r['target'] for r in partition])))
        for epoch in range(self.start_epoch, self.epochs):
            for split, (values, targets) in enumerate(prepared):
                self.net.train(split == 0)
                order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(self.seed + epoch)) if split == 0 else torch.arange(len(targets))
                total_loss = correct = seen = 0
                for offset in range(0, len(order), self.batch_size):
                    if not self.check_control():
                        return
                    ids = order[offset:offset + self.batch_size]
                    with torch.set_grad_enabled(split == 0):
                        logits = self.net(values[ids])
                        loss = nn.functional.cross_entropy(logits, targets[ids])
                        if split == 0:
                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            self.optimizer.step()
                            self.step += 1
                    seen += len(ids)
                    total_loss += loss.item() * len(ids)
                    correct += int((logits.argmax(-1) == targets[ids]).sum())
                    prefix = 'validation_' if split else ''
                    self.report({'phase': 'validation' if split else 'training', 'epoch': epoch + 1,
                                 'total_epochs': self.epochs, 'step': self.step,
                                 prefix + 'loss': total_loss / seen, prefix + 'accuracy': correct / seen,
                                 prefix + 'progress': 100. * seen / len(targets)})
            self.start_epoch = epoch + 1
            self.save_checkpoint(self.run_dir / 'checkpoints' / f'checkpoint_{self.start_epoch:04d}.pt')

    def save_checkpoint(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': self.net.state_dict(), 'model_settings': {'hidden_size': self.net.hidden_size},
                    'feature_version': FEATURE_VERSION, 'reference': self.reference, 'settings': self.settings,
                    'mmseqs_version': self.mmseqs_version, 'dataset_digest': self.dataset_digest,
                    'optimizer': self.optimizer.state_dict(), 'epoch': self.start_epoch, 'step': self.step,
                    'metrics': self.get_metrics(), 'config': self.config}, path)
        path.with_suffix('.json').write_text(json.dumps({'epoch': self.start_epoch, 'step': self.step,
                                                        'models': self.config.get('models', {}), **self.get_metrics()}, indent=2))

    def load_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['feature_version'] != FEATURE_VERSION or state['settings'] != self.settings:
            raise ValueError('Keep the checkpoint feature, cluster and search settings when resuming')
        if state['model_settings'] != {'hidden_size': self.net.hidden_size}:
            raise ValueError('Checkpoint architecture differs from selected model')
        self.net.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.reference = state['reference']
        self.dataset_digest = state['dataset_digest']
        self.mmseqs_version = state['mmseqs_version']
        self.start_epoch, self.step = state['epoch'], state['step']
        self._metrics.update(state.get('metrics', {}))
