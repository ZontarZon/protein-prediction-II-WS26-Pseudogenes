"""Train an MLP head on frozen ProtT5 embeddings using prepared sequence splits.

Run from the repository root with:
    python -m models.ModelsAndTrainers.Trainers.PLM_MLPTrainer
"""
import argparse
import math
from pathlib import Path

import torch
from torch import nn

from models.ModelsAndTrainers.Models.ProteinPEClassifier import CHECKPOINT, ProteinPEClassifier
from models.ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import load_prepared


# Defaults follow the head search in docs/mlp_experiments/report.html.
hparams = {
    'training_data': 'dataset/prepared.pt',
    'run_dir': 'runs/plm_mlp',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'epochs': 6,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 0.0,
    'seed': 7,
    'hidden_sizes': [512, 512],
    'dropout': 0.5,
    'norm': 'layernorm',
    'activation': 'gelu',
}


class PLM_MLPTrainer:
    def __init__(self, config=None):
        self.config = {**hparams, **(config or {})}
        for key in ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'seed'):
            setattr(self, key, self.config[key])
        for key in ('epochs', 'batch_size', 'seed'):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == 'seed' else 1):
                raise ValueError(f'{key} must be a valid integer')
        if type(self.learning_rate) not in (int, float) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if type(self.weight_decay) not in (int, float) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        self.device = torch.device(self.config['device'])
        self.run_dir = Path(self.config['run_dir'])
        torch.manual_seed(self.seed)
        self.net = ProteinPEClassifier(**{
            key: self.config[key] for key in ('hidden_sizes', 'dropout', 'norm', 'activation')
        }).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.net.head.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.settings = self.prepared_digest = self.dataset_digest = self.mmseqs_version = None
        self.start_epoch = self.step = 0
        self.metrics = {}

    def train(self):
        state, digest = load_prepared(self.config['training_data'])
        if self.prepared_digest is not None and self.prepared_digest != digest:
            raise ValueError('Preprocessed dataset changed since checkpoint')
        if self.dataset_digest is not None and self.dataset_digest != state['dataset_digest']:
            raise ValueError('Dataset changed since checkpoint')
        if self.settings is not None and self.settings != state['settings']:
            raise ValueError('Preprocessing settings changed since checkpoint')
        if self.mmseqs_version is not None and self.mmseqs_version != state['mmseqs_version']:
            raise ValueError('Preprocessing MMseqs2 version changed since checkpoint')
        self.prepared_digest = digest
        self.dataset_digest = state['dataset_digest']
        self.settings = state['settings']
        self.mmseqs_version = state['mmseqs_version']
        prepared = [(state[split]['records'], state[split]['targets']) for split in ('train', 'validation')]
        self.net.setup_encoder()
        for epoch in range(self.start_epoch, self.epochs):
            for split, (values, targets) in enumerate(prepared):
                self.net.train(split == 0)
                order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(self.seed + epoch)) if split == 0 else torch.arange(len(targets))
                total_loss = correct = seen = 0
                for offset in range(0, len(order), self.batch_size):
                    ids = order[offset:offset + self.batch_size]
                    with torch.set_grad_enabled(split == 0):
                        sequences = [values[index]['sequence'] for index in ids.tolist()]
                        labels = targets[ids].to(self.device)
                        logits = self.net(sequences)
                        loss = nn.functional.cross_entropy(logits, labels)
                        if split == 0:
                            self.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            self.optimizer.step()
                            self.step += 1
                    seen += len(ids)
                    total_loss += loss.item() * len(ids)
                    correct += int((logits.argmax(-1) == labels).sum())
                prefix = 'validation_' if split else ''
                self.metrics.update({prefix + 'loss': total_loss / seen,
                                     prefix + 'accuracy': correct / seen})
                print(f"Epoch {epoch + 1}/{self.epochs} "
                      f"{'validation' if split else 'train'}: "
                      f"loss={total_loss / seen:.4f}, accuracy={correct / seen:.4f}", flush=True)
            self.start_epoch = epoch + 1
            self.save_checkpoint(self.run_dir / 'checkpoints' / f'checkpoint_{self.start_epoch:04d}.pt')

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'pipeline': 'mmseqs2_prott5', 'encoder_checkpoint': CHECKPOINT,
                    'rng_state': torch.get_rng_state(), 'shuffle_seed': self.seed,
                    'cuda_rng_state': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                    'model_state_dict': {f'head.{k}': v for k, v in self.net.head.state_dict().items()}, 'model_settings': self.net.model_settings(),
                    'settings': self.settings,
                    'mmseqs_version': self.mmseqs_version, 'dataset_digest': self.dataset_digest,
                    'prepared_digest': self.prepared_digest, 'optimizer': self.optimizer.state_dict(), 'epoch': self.start_epoch, 'step': self.step,
                    'metrics': self.metrics, 'config': self.config}, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('pipeline') != 'mmseqs2_prott5':
            raise ValueError('Legacy 11-feature checkpoints cannot resume sequence-based training')
        if state['shuffle_seed'] != self.seed:
            raise ValueError('Keep the training shuffle seed when resuming')
        if state['model_settings'] != self.net.model_settings():
            raise ValueError('Checkpoint architecture differs from selected model')
        self.net.load_weight_state(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.settings = state['settings']
        self.prepared_digest = state.get('prepared_digest')
        self.dataset_digest = state['dataset_digest']
        self.mmseqs_version = state['mmseqs_version']
        self.start_epoch, self.step = state['epoch'], state['step']
        self.metrics.update(state.get('metrics', {}))
        torch.set_rng_state(state['rng_state'].cpu())
        if self.device.type == 'cuda' and state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-data', default=hparams['training_data'])
    parser.add_argument('--run-dir', default=hparams['run_dir'])
    parser.add_argument('--device', default=hparams['device'])
    parser.add_argument('--resume', type=Path, help='Resume a saved training checkpoint')
    args = parser.parse_args()
    trainer = PLM_MLPTrainer({
        'training_data': args.training_data, 'run_dir': args.run_dir, 'device': args.device,
    })
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
