from pathlib import Path
from typing import Any

import torch
from torch import nn


class DivisionMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, int] = (64, 32),
        dropout_1: float = 0.10,
        dropout_2: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.SiLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def load_division_checkpoint(
    path: Path,
    map_location: str | torch.device = 'cpu',
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    source_hidden_raw = tuple(int(value) for value in payload.get('source_hidden', (96, 48)))
    pair_hidden_raw = tuple(int(value) for value in payload.get('pair_hidden', (128, 64)))
    source_hidden = (source_hidden_raw[0], source_hidden_raw[1])
    pair_hidden = (pair_hidden_raw[0], pair_hidden_raw[1])
    dropout_1 = float(payload.get('dropout_1', 0.10))
    dropout_2 = float(payload.get('dropout_2', 0.05))
    source_features = int(payload['source_model']['net.0.weight'].shape[1])
    pair_features = int(payload['pair_model']['net.0.weight'].shape[1])
    source_model = DivisionMLP(source_features, source_hidden, dropout_1, dropout_2)
    pair_model = DivisionMLP(pair_features, pair_hidden, dropout_1, dropout_2)
    source_model.load_state_dict(payload['source_model'])
    pair_model.load_state_dict(payload['pair_model'])
    source_model.eval()
    pair_model.eval()
    return {
        'source_model': source_model,
        'pair_model': pair_model,
        'payload': payload,
    }
