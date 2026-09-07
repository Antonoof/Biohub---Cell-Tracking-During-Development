from pathlib import Path

import numpy as np
import torch

from biohub.models.option_head import OptionHead


def grouped_best(owner: np.ndarray, score: np.ndarray, n_sources: int) -> np.ndarray:
    result = np.full(n_sources, -1, np.int64)
    order = np.argsort(owner, kind='stable')
    if not len(order):
        return result
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for left, right in zip(starts, ends):
        rows = order[left:right]
        result[int(sorted_owner[left])] = int(rows[int(np.argmax(score[rows]))])
    return result


def cap_pair_options(owner: np.ndarray, max_pairs: int) -> np.ndarray:
    order = np.argsort(owner, kind='stable')
    if not len(order):
        return order
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    keep = np.concatenate(
        [order[left : min(right, left + max_pairs)] for left, right in zip(starts, ends)]
    )
    return np.sort(keep)


class SourceCardinalityRuntime:
    def __init__(self, artifact_dir: str | Path):
        artifact_dir = Path(artifact_dir)
        payload = torch.load(
            artifact_dir / 'source_cardinality_head.pt',
            map_location='cpu',
            weights_only=False,
        )
        self.config = dict(payload['config'])
        self.threshold = float(self.config['threshold'])
        self.model = OptionHead(
            int(self.config['source_dim']),
            int(self.config['pair_dim']),
            int(self.config['hidden_source']),
            int(self.config['hidden_pair']),
        )
        self.model.load_state_dict(payload['state_dict'], strict=True)
        self.model.eval()
        self.source_mean = np.asarray(payload['source_mean'], np.float32)
        self.source_scale = np.asarray(payload['source_scale'], np.float32)
        self.pair_mean = np.asarray(payload['pair_mean'], np.float32)
        self.pair_scale = np.asarray(payload['pair_scale'], np.float32)

    @torch.no_grad()
    def score(
        self,
        source_features: np.ndarray,
        pair_features: np.ndarray,
        pair_source_row: np.ndarray,
        pair_nodes: np.ndarray,
        batch_size: int = 250_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.nan_to_num(np.asarray(source_features, np.float32))
        pair = np.nan_to_num(np.asarray(pair_features, np.float32))
        owner = np.asarray(pair_source_row, np.int32)
        nodes = np.asarray(pair_nodes, np.int64)
        if source.ndim != 2 or source.shape[1] != len(self.source_mean):
            raise ValueError(f'Source feature mismatch: {source.shape}')
        if pair.ndim != 2 or pair.shape[1] != len(self.pair_mean):
            raise ValueError(f'Pair feature mismatch: {pair.shape}')
        if len(pair) != len(owner) or nodes.shape != (len(pair), 2):
            raise ValueError('Pair owner/node alignment mismatch')
        if len(owner) and (owner.min() < 0 or owner.max() >= len(source)):
            raise ValueError('Pair source row is out of bounds')
        retained = cap_pair_options(owner, int(self.config['max_pairs']))
        pair = pair[retained]
        owner = owner[retained]
        nodes = nodes[retained]

        source_norm = np.clip((source - self.source_mean) / self.source_scale, -10.0, 10.0)
        source_h = self.model.source_encoder(torch.from_numpy(source_norm)).cpu()
        continue_logit = self.model.continue_head(source_h).squeeze(-1).numpy()
        pair_logits = np.empty(len(pair), np.float32)
        for left in range(0, len(pair), batch_size):
            right = min(len(pair), left + batch_size)
            block = np.clip(
                (pair[left:right] - self.pair_mean) / self.pair_scale,
                -10.0,
                10.0,
            )
            pair_h = self.model.pair_encoder(torch.from_numpy(block))
            source_block = source_h[torch.from_numpy(owner[left:right].astype(np.int64))]
            value = self.model.divide_head(torch.cat([source_block, pair_h], dim=1)).squeeze(-1)
            pair_logits[left:right] = (value + self.model.division_bias).cpu().numpy()

        n_sources = len(source)
        maximum = np.full(n_sources, -np.inf, np.float32)
        np.maximum.at(maximum, owner, pair_logits)
        total = np.zeros(n_sources, np.float64)
        valid_pair = np.isfinite(maximum[owner])
        np.add.at(
            total,
            owner[valid_pair],
            np.exp(pair_logits[valid_pair] - maximum[owner[valid_pair]]),
        )
        valid_source = total > 0
        log_total = np.full(n_sources, -np.inf, np.float32)
        log_total[valid_source] = maximum[valid_source] + np.log(total[valid_source]).astype(
            np.float32
        )
        delta = np.clip(log_total - continue_logit, -40.0, 40.0)
        source_probability = np.zeros(n_sources, np.float32)
        source_probability[valid_source] = 1.0 / (1.0 + np.exp(-delta[valid_source]))
        best = grouped_best(owner, pair_logits, n_sources)
        best_nodes = np.full((n_sources, 2), -1, np.int64)
        valid_best = best >= 0
        best_nodes[valid_best] = nodes[best[valid_best]]
        return source_probability, best_nodes
