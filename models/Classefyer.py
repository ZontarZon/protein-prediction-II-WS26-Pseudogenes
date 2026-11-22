from torch import nn


class Classefyer(nn.Sequential):
    """Map a batch of embeddings to regular and pseudogene logits."""

    def __init__(self, embedding_dim, hidden_sizes):
        layers = []
        input_size = embedding_dim
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 2))
        super().__init__(*layers)


class GELUClassefyer(nn.Module):
    """Map a batch of embeddings to two logits using a deeper MLP."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 2),
        )

    def forward(self, embeddings):
        return self.layers(embeddings)
