import json
import random

import torch
from torch.utils.data import TensorDataset

CLASS_NAMES = ("regular", "pseudogene")
LABELS = {tag: index for index, tag in enumerate(CLASS_NAMES)}


def load_splits(path, val_fraction, test_fraction, seed):
    """Shuffle whole clusters, then flatten the items inside each split."""
    if not (0 < val_fraction and 0 < test_fraction and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")

    with open(path, encoding="utf-8") as file:
        clusters = json.load(file)

    random.Random(seed).shuffle(clusters)
    val_count = max(1, round(len(clusters) * val_fraction))
    test_count = max(1, round(len(clusters) * test_fraction))
    train_end = len(clusters) - val_count - test_count
    if train_end < 1:
        raise ValueError("Split fractions must leave at least one cluster for training.")
    val_end = train_end + val_count

    splits = []
    for group in (clusters[:train_end], clusters[train_end:val_end], clusters[val_end:]):
        items = [item for cluster in group for item in cluster["items"]]
        if not items:
            raise ValueError("Each split needs at least one item. Check the data and split fractions.")
        embeddings = torch.tensor([item["embedding"] for item in items], dtype=torch.float32)
        labels = torch.tensor([LABELS[item["tag"]] for item in items], dtype=torch.long)
        splits.append(TensorDataset(embeddings, labels))

    return tuple(splits)
