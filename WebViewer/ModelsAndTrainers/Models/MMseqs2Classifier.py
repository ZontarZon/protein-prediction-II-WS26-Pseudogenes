"""Trainable classifier over MMseqs2 homology features; no encoder downloads."""
from pathlib import Path
import tempfile

import torch
from torch import nn

from backend.models.base import ModelBase
from ModelsAndTrainers.DataLoaders.MMseqs2Runner import MMseqs2Runner
from ModelsAndTrainers.DataLoaders.MMseqs2ClassificationData import (
    FEATURE_COUNT, FEATURE_VERSION, features, protein_sequence,
)


class MMseqs2Classifier(ModelBase):
    model_id = 'MMseqs2Classifier'
    display_name = 'MMseqs2 PE classifier'
    description = 'Homology features and a binary head for UniProt PE 1–4 versus PE 5 evidence labels.'
    category = 'protein classification'
    interfaces = ['mmseqs2_pe_classifier']
    inputs = [{'key': 'sequence', 'type': 'text', 'label': 'Protein sequence or single FASTA entry'}]
    output = {'type': 'classification', 'label': 'PE evidence probabilities'}
    inference_parameters = [
        {'key': 'sequence', 'type': 'text', 'label': 'Protein sequence or FASTA entry', 'required': True},
        {'key': 'executable', 'type': 'text', 'label': 'MMseqs2 executable', 'default': 'mmseqs'},
    ]

    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = int(hidden_size)
        if self.hidden_size < 1:
            raise ValueError('hidden_size must be positive')
        self.head = nn.Sequential(nn.Linear(FEATURE_COUNT, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 2))

    def forward(self, values):
        return self.head(values)

    def prepare_inference(self, inputs, checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['feature_version'] != FEATURE_VERSION:
            raise ValueError('Unsupported MMseqs2 feature version')
        self.__init__(**state['model_settings'])
        self.load_state_dict(state['model_state_dict'])
        self.reference = state['reference']
        if not self.reference:
            raise ValueError('Checkpoint was saved before reference preparation; resume training first')
        self.search_settings = state['settings']
        self.runner = MMseqs2Runner(inputs.get('executable') or 'mmseqs')
        if self.runner.version != state['mmseqs_version']:
            raise ValueError('Inference requires the MMseqs2 version recorded in this checkpoint')
        previous = getattr(self, '_search_cache', None)
        if previous is not None:
            previous.cleanup()
        self._search_cache = tempfile.TemporaryDirectory(prefix='mmseqs2-inference-')
        self.eval()

    def infer(self, inputs, checkpoint, progress_callback=None):
        self.prepare_inference(inputs, checkpoint)
        try:
            return self.infer_prepared(inputs, progress_callback)
        finally:
            self._search_cache.cleanup()

    def infer_prepared(self, inputs, progress_callback=None):
        sequence = protein_sequence(inputs.get('sequence', ''))
        queries = [{'key': 'query', 'sequence': sequence}]
        if progress_callback:
            progress_callback(0, 1)
        hits = self.runner.search(queries, self.reference, Path(self._search_cache.name),
                                  self.search_settings['sensitivity'], self.search_settings['evalue'])
        values = features(queries, self.reference, hits)
        with torch.no_grad():
            probabilities = self(values).softmax(-1)[0].tolist()
        if progress_callback:
            progress_callback(1, 1)
        return {'classes': [{'label': 'PE 1–4', 'probability': probabilities[0]},
                            {'label': 'PE 5', 'probability': probabilities[1]}],
                'residues': len(sequence), 'no_hits': bool(values[0, 10]), 'execution_device': 'cpu',
                'note': 'PE labels describe existence evidence, not pseudogene status. Probabilities are not calibrated certainty.'}
