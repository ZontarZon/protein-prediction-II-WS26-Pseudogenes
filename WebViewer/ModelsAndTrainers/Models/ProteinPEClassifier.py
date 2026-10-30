"""Frozen ProtT5 protein embeddings with a trainable binary classification head."""
import re
import gc
from contextlib import nullcontext

import torch
from torch import nn

from backend.models.base import ModelBase

CHECKPOINT = 'Rostlab/prot_t5_xl_half_uniref50-enc'


def sequence_input(value):
    lines = str(value).strip().splitlines()
    if lines and lines[0].startswith('>'):
        lines = lines[1:]
    if any(line.startswith('>') for line in lines):
        raise ValueError('Enter exactly one protein sequence or FASTA entry')
    sequence = ''.join(''.join(lines).split()).upper()
    if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWYXBZOUJ'):
        raise ValueError('Enter a nonempty protein sequence using amino-acid letters')
    return re.sub('[UZOBJ]', 'X', sequence)


class ProteinPEClassifier(ModelBase):
    model_id = 'ProteinPEClassifier'
    display_name = 'ProtT5 PE classifier'
    description = 'Frozen protein embeddings and a binary head: PE 1–4 (Real), PE 5 (Fake). These labels describe UniProt existence evidence, not biological authenticity.'
    category = 'protein classification'
    inputs = [{'key': 'sequence', 'type': 'text', 'label': 'Protein sequence or single FASTA entry'}]
    output = {'type': 'classification', 'label': 'Real / Fake probabilities'}
    interfaces = ['protein_pe_classifier']
    inference_parameters = [{'key': 'sequence', 'type': 'text', 'label': 'Protein sequence or FASTA entry', 'required': True},
                            {'key': 'device', 'type': 'selection', 'label': 'Device', 'options': ['cpu', 'cuda'], 'default': 'cpu'},
                            {'key': 'explain', 'type': 'boolean', 'label': 'Residue contribution heat map (slower inference)', 'default': False}]

    def __init__(self, hidden_size=256, max_residues=None):
        super().__init__()
        # max_residues is accepted only for compatibility with older checkpoints.
        self.hidden_size = int(hidden_size)
        if self.hidden_size < 1:
            raise ValueError('hidden_size must be positive')
        self.head = nn.Sequential(nn.Linear(1024, self.hidden_size), nn.ReLU(), nn.Dropout(0.1), nn.Linear(self.hidden_size, 2))
        self.encoder = None
        self.tokenizer = None

    def setup_encoder(self):
        if self.encoder is not None:
            return
        from transformers import T5EncoderModel, T5Tokenizer
        from transformers.utils import logging as transformer_logging
        from huggingface_hub.utils import disable_progress_bars
        # Keep warnings/errors; send training progress through TrainerBase.report.
        transformer_logging.set_verbosity_warning()
        transformer_logging.disable_progress_bar()
        disable_progress_bars()
        device = next(self.head.parameters()).device
        self.tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT, do_lower_case=False)
        self.encoder = T5EncoderModel.from_pretrained(CHECKPOINT, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def embed(self, sequences):
        sequences = [sequence_input(s) for s in sequences]
        self.setup_encoder()
        device = next(self.head.parameters()).device
        # Bound attention memory while including every residue. Each chunk is
        # encoded separately, then weighted by residue count for a protein mean.
        embeddings = []
        with torch.no_grad():
            for sequence in sequences:
                total = torch.zeros(1024, device=device, dtype=torch.float32)
                for start in range(0, len(sequence), 1024):
                    chunk = sequence[start:start + 1024]
                    tokens = self.tokenizer([' '.join(chunk)], return_tensors='pt', padding=True)
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    hidden = self.encoder(**tokens).last_hidden_state
                    total += hidden[0, :len(chunk)].float().sum(0)
                embeddings.append(total / len(sequence))
        return torch.stack(embeddings)

    def forward(self, sequences):
        return self.head(self.embed(sequences))

    def explain_sequence(self, sequence, steps=32, progress_callback=None):
        """Inference-only embedding Integrated Gradients; never populate parameter .grad.

        Recompute each chunk's graph separately, using the head gradient at the
        full interpolated protein mean. Thus memory does not scale with chunks.
        """
        if self.training:
            raise RuntimeError('Sequence explanations require evaluation mode')
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('Integration steps must be positive')
        sequence = sequence_input(sequence)
        self.setup_encoder()
        device = next(self.head.parameters()).device
        chunks = []
        with torch.no_grad():
            for start in range(0, len(sequence), 1024):
                chunk = sequence[start:start + 1024]
                tokens = {k: v.to(device) for k, v in self.tokenizer(
                    [' '.join(chunk)], return_tensors='pt', padding=True).items()}
                reference = self.tokenizer([' '.join('X' * len(chunk))], return_tensors='pt', padding=True)
                reference_ids = reference['input_ids'].to(device)
                if reference_ids.shape != tokens['input_ids'].shape:
                    raise ValueError('Reference token alignment differs from the sequence')
                actual = self.encoder.get_input_embeddings()(tokens['input_ids']).detach()
                baseline = self.encoder.get_input_embeddings()(reference_ids).detach()
                chunks.append((len(chunk), tokens['attention_mask'], baseline, actual - baseline))

        def pooled(alpha):
            total = torch.zeros(1, 1024, device=device)
            for length, mask, baseline, delta in chunks:
                hidden = self.encoder(inputs_embeds=baseline + alpha * delta,
                                      attention_mask=mask).last_hidden_state
                total += hidden[:, :length].float().sum(1) / len(sequence)
            return total

        scores = [torch.zeros(length, device=device) for length, *_ in chunks]
        with torch.no_grad():
            baseline_logits = self.head(pooled(0))[0]
            actual_logits = self.head(pooled(1))[0]
            baseline_score = float(baseline_logits[0] - baseline_logits[1])
            actual_score = float(actual_logits[0] - actual_logits[1])
        for index in range(steps):
            alpha = (index + .5) / steps
            with torch.no_grad():
                mean = pooled(alpha)
            # Only explanation gradients use offloaded activations. The encoder
            # weights and the ordinary training/embedding path stay unchanged.
            storage = torch.autograd.graph.save_on_cpu(pin_memory=True) if device.type == 'cuda' else nullcontext()
            with torch.enable_grad(), storage:
                mean = mean.detach().requires_grad_(True)
                logits = self.head(mean)[0]
                head_gradient, = torch.autograd.grad(logits[0] - logits[1], mean)
                for chunk_index, (length, mask, baseline, delta) in enumerate(chunks):
                    embedded = (baseline + alpha * delta).detach().requires_grad_(True)
                    hidden = self.encoder(inputs_embeds=embedded, attention_mask=mask).last_hidden_state
                    contribution = hidden[:, :length].float().sum(1) / len(sequence)
                    gradient, = torch.autograd.grad(contribution, embedded, grad_outputs=head_gradient)
                    scores[chunk_index] += (gradient.float() * delta.float()).sum(-1)[0, :length].detach() / steps
                    del hidden, contribution, gradient, embedded
            if progress_callback:
                progress_callback(index + 1, steps + 1)
        values = torch.cat(scores).cpu().tolist()
        if not torch.isfinite(torch.tensor(values)).all().item():
            raise ValueError('Non-finite residue attribution; try inference on CPU')
        return {'sequence': sequence, 'scores': values, 'method': 'Integrated Gradients',
                'baseline': 'X at every residue', 'steps': steps, 'chunk_size': 1024,
                'score': actual_score, 'baseline_score': baseline_score,
                'completeness_error': sum(values) - (actual_score - baseline_score)}

    def freeze_weights(self, frozen=True):
        self.head.requires_grad_(not frozen)
        if self.encoder is not None:
            self.encoder.requires_grad_(False)

    def extract_weight_state(self, checkpoint_state):
        return super().extract_weight_state(checkpoint_state)

    def load_weight_state(self, state, strict=True):
        return self.head.load_state_dict({k.removeprefix('head.'): v for k, v in state.items()}, strict=strict)

    def prepare_inference(self, inputs, checkpoint):
        """Load once for a processing run; training never calls this method."""
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        settings = state.get('model_settings', {})
        if settings:
            self.__init__(**settings)
        self.load_weight_state(state.get('model_state_dict', state))
        device = inputs.get('device', 'cpu')
        self.to(device).eval()
        self.setup_encoder()
        self._inference_fallback = None

    def infer(self, inputs, checkpoint, progress_callback=None):
        inputs = {**inputs, 'sequence': sequence_input(inputs.get('sequence', ''))}
        if progress_callback:
            progress_callback(0, 33 if inputs.get('explain') is True else 1)
        self.prepare_inference(inputs, checkpoint)
        return self.infer_prepared(inputs, progress_callback)

    def infer_prepared(self, inputs, progress_callback=None):
        """Predict with an already loaded inference model."""
        if self.training:
            raise RuntimeError('Prepared inference requires evaluation mode')
        try:
            return self._predict_sequence(inputs, progress_callback)
        except torch.cuda.OutOfMemoryError:
            if not self._using_cuda():
                raise
        # Leave the exception scope first: its traceback can retain failed
        # autograd graphs. Keep this job on CPU rather than failing every long
        # record again. Widen the half-precision encoder for CPU computation.
        gc.collect()
        self.to(device='cpu', dtype=torch.float32)
        torch.cuda.empty_cache()
        self._inference_fallback = 'cuda_out_of_memory'
        return self._predict_sequence(inputs, progress_callback)

    def _using_cuda(self):
        return next(self.head.parameters()).device.type == 'cuda'

    def _predict_sequence(self, inputs, progress_callback=None):
        sequence = sequence_input(inputs.get('sequence', ''))
        total_steps = 33 if inputs.get('explain') is True else 1
        if progress_callback:
            progress_callback(0, total_steps)
        with torch.no_grad():
            probabilities = self([sequence]).softmax(-1)[0].cpu().tolist()
        explanation = self.explain_sequence(sequence, progress_callback=progress_callback) if inputs.get('explain') is True else None
        if progress_callback:
            progress_callback(total_steps, total_steps)
        return {'classes': [{'label': 'Class 1 (Real value)', 'probability': probabilities[0]},
                            {'label': 'Class 2 (Fake value)', 'probability': probabilities[1]}],
                **({'attribution': explanation} if explanation is not None else {}),
                'execution_device': str(next(self.head.parameters()).device),
                **({'memory_fallback': self._inference_fallback} if getattr(self, '_inference_fallback', None) else {}),
                'residues': len(sequence), 'note': 'Model probabilities are not calibrated certainty. Real = PE 1–4; Fake = PE 5 (uncertain existence).'}
