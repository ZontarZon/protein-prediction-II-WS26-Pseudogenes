"""ProtT5 classifier trained on sequence partitions prepared with MMseqs2."""
import torch

from ModelsAndTrainers.Models.ProteinPEClassifier import ProteinPEClassifier


class MMseqs2Classifier(ProteinPEClassifier):
    model_id = 'MMseqs2Classifier'
    display_name = 'MMseqs2 split · ProtT5 PE classifier'
    description = 'Frozen ProtT5 sequence embeddings and a trainable head, using MMseqs2 cluster-based splits.'
    interfaces = ['mmseqs2_pe_classifier']

    def prepare_inference(self, inputs, checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('pipeline') != 'mmseqs2_prott5':
            raise ValueError('Legacy homology-feature checkpoints are incompatible; train a new sequence-based classifier.')
        super().prepare_inference(inputs, checkpoint)
