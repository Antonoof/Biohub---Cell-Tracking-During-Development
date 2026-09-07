import torch
from torch import nn


class OptionHead(nn.Module):
    def __init__(self, source_dim: int, pair_dim: int, hidden_source: int, hidden_pair: int):
        super().__init__()
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, hidden_source),
            nn.SiLU(),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_pair),
            nn.SiLU(),
        )
        self.continue_head = nn.Linear(hidden_source, 1)
        self.divide_head = nn.Linear(hidden_source + hidden_pair, 1)
        self.division_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        source_x: torch.Tensor,
        pair_x: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_h = self.source_encoder(source_x)
        pair_h = self.pair_encoder(pair_x)
        expanded = source_h[:, None, :].expand(-1, pair_h.shape[1], -1)
        pair_logits = self.divide_head(torch.cat([expanded, pair_h], dim=-1)).squeeze(-1)
        pair_logits = pair_logits + self.division_bias
        pair_logits = pair_logits.masked_fill(~pair_mask, -1e9)
        continue_logit = self.continue_head(source_h).squeeze(-1)
        return continue_logit, pair_logits
