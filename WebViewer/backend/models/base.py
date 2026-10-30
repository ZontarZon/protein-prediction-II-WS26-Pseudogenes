from abc import ABC
from torch import nn
import torch


class ModelBase(nn.Module, ABC):
    model_id = ""
    display_name = ""
    description = ""
    category = "general"
    inputs = []
    output = {"type": "numerical", "label": "Model output"}
    interfaces = []
    inference_parameters = []
    checkpoint_required = True
    supports_freezing = True

    def __init__(self):
        super().__init__()

    @classmethod
    def metadata(cls):
        return {"id": cls.model_id or cls.__name__, "name": cls.display_name or cls.__name__,
                "description": cls.description, "category": cls.category,
                "inputs": cls.inputs, "output": cls.output, "interfaces": cls.interfaces,
                "inference_parameters": cls.inference_parameters,
                "checkpoint_required": cls.checkpoint_required, "inference_available": cls.has_inference(),
                "supports_freezing": cls.supports_freezing, "weight_capable": True, "weight_format": "state_dict"}

    def extract_weight_state(self, checkpoint_state):
        if not isinstance(checkpoint_state, dict):
            raise ValueError("Checkpoint does not contain a model state dictionary")
        for key in ("model_state_dict", "state_dict", "weights"):
            state = checkpoint_state.get(key)
            if isinstance(state, dict) and state and all(torch.is_tensor(value) for value in state.values()):
                return {name: value.detach().cpu() for name, value in state.items()}
        raise ValueError(f"{type(self).__name__} cannot extract model weights from this checkpoint format")

    def load_weight_state(self, state, strict=True):
        return self.load_state_dict(state, strict=strict)

    def freeze_weights(self, frozen=True):
        if not self.supports_freezing:
            raise ValueError(f"{type(self).__name__} does not support freezing")
        for parameter in self.parameters():
            parameter.requires_grad_(not frozen)

    @classmethod
    def has_inference(cls):
        return cls.infer is not ModelBase.infer

    def infer(self, inputs, checkpoint: str, progress_callback=None):
        raise NotImplementedError(f"{type(self).__name__} does not implement inference")
