from torch import nn


class DivisionMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)
