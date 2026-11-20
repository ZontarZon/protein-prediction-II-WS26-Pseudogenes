from torch import nn


class Classefyer(nn.Sequential):
    """Map a batch of embeddings to regular and pseudogene logits."""

    def __init__(self, embedding_dim, hidden_sizes, dropout=0.0):
        layers = []
        input_size = embedding_dim
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 2))
        super().__init__(*layers)
