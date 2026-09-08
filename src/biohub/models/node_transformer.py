import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(dtype=x.dtype) * self.weight


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x))))


def make_ffn(hidden_dim: int, mlp_ratio: float, dropout: float, act: str) -> nn.Module:
    hidden = int(hidden_dim * mlp_ratio)
    if act == 'swiglu':
        return SwiGLU(hidden_dim, hidden, dropout)
    activation: nn.Module = nn.GELU() if act == 'gelu' else nn.SiLU()
    if act not in ('gelu', 'silu'):
        raise ValueError(f'Unknown ffn_act {act!r}')
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden),
        activation,
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden_dim),
        nn.Dropout(dropout),
    )


def pair_geom_dim(kind: str) -> int:
    if kind == 'rel':
        return 3
    if kind == 'dist':
        return 1
    if kind == 'rel_dist':
        return 4
    raise ValueError(f'Unknown pair_geom {kind!r}')


class BilinearPairHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, geom_dim: int = 3) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.rel = nn.Linear(geom_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        return self.drop(self.bilinear(query, key) + self.rel(rel)).squeeze(-1)


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == 'rmsnorm':
        return RMSNorm(dim)
    if kind == 'layernorm':
        return nn.LayerNorm(dim)
    raise ValueError(f'Unknown norm {kind!r}')


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float | None = None,
        drop_path: float = 0.0,
        norm: str = 'layernorm',
        layer_scale_init: float = 0.0,
        ffn_act: str = 'gelu',
    ):
        super().__init__()
        self.norm1 = make_norm(norm, hidden_dim)
        self.norm2 = make_norm(norm, hidden_dim)
        attn_p = dropout if attn_dropout is None else attn_dropout
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=attn_p
        )
        self.mlp = make_ffn(hidden_dim, mlp_ratio, dropout, ffn_act)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if layer_scale_init > 0:
            self.ls1: nn.Module = LayerScale(hidden_dim, layer_scale_init)
            self.ls2: nn.Module = LayerScale(hidden_dim, layer_scale_init)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.shape[1] == 0:
            return q
        if kv.shape[1] == 0:
            return q + self.drop_path(self.ls2(self.mlp(self.norm2(q))))
        key_padding_mask = None
        blocked = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask
            blocked = key_padding_mask.all(dim=-1)
            if blocked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[blocked, 0] = False
        attn_out, _ = self.cross_attn(
            self.norm1(q),
            self.norm1(kv),
            self.norm1(kv),
            key_padding_mask=key_padding_mask,
        )
        if blocked is not None and blocked.any():
            attn_out = attn_out.masked_fill(blocked[:, None, None], 0)
        q = q + self.drop_path(self.ls1(attn_out))
        q = q + self.drop_path(self.ls2(self.mlp(self.norm2(q))))
        return q


class SimpleNodeTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int = 33,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
        drop_path: float = 0.0,
        use_self_attn: bool = False,
        norm: str = 'layernorm',
        rel_coord_scale: float = 100.0,
        pair_head: str = 'mlp',
        layer_scale_init: float = 0.0,
        gradient_checkpointing: bool = False,
        ffn_act: str = 'gelu',
        attn_dropout: float | None = None,
        drop_path_decay: bool = False,
        pair_geom: str = 'rel',
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.rel_coord_scale = float(rel_coord_scale)
        self.pair_head_kind = pair_head
        self.pair_geom = pair_geom
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = make_norm(norm, hidden_dim)
        if drop_path_decay and n_blocks > 1:
            rates = [drop_path * i / (n_blocks - 1) for i in range(n_blocks)]
        else:
            rates = [drop_path] * n_blocks
        attn_p = dropout if attn_dropout is None else attn_dropout
        self.self_blocks = (
            nn.ModuleList(
                [
                    CrossAttentionBlock(
                        hidden_dim=hidden_dim,
                        n_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_dropout=attn_p,
                        drop_path=rate,
                        norm=norm,
                        layer_scale_init=layer_scale_init,
                        ffn_act=ffn_act,
                    )
                    for rate in rates
                ]
            )
            if use_self_attn
            else None
        )
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_p,
                    drop_path=rate,
                    norm=norm,
                    layer_scale_init=layer_scale_init,
                    ffn_act=ffn_act,
                )
                for rate in rates
            ]
        )
        self.norm_out = make_norm(norm, hidden_dim)
        geom_dim = pair_geom_dim(pair_geom)
        if pair_head == 'bilinear':
            self.pair_mlp: nn.Module = BilinearPairHead(hidden_dim, dropout, geom_dim)
        elif pair_head == 'mlp':
            self.pair_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + geom_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            raise ValueError(f'Unknown pair_head {pair_head!r}')

    def _pair_geom(self, rel: torch.Tensor) -> torch.Tensor:
        dist = rel.norm(dim=-1, keepdim=True)
        if self.pair_geom == 'dist':
            return dist
        if self.pair_geom == 'rel_dist':
            return torch.cat([rel, dist], dim=-1)
        return rel

    def _pair_scores(
        self,
        qc: torch.Tensor,
        kk: torch.Tensor,
        cc: torch.Tensor,
        cc1: torch.Tensor,
    ) -> torch.Tensor:
        nc_i = qc.shape[1]
        n1 = kk.shape[1]
        qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
        ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
        rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / self.rel_coord_scale
        geom = self._pair_geom(rel)
        if isinstance(self.pair_mlp, BilinearPairHead):
            return self.pair_mlp(qe, ke, geom)
        return self.pair_mlp(torch.cat([qe, ke, geom], dim=-1)).squeeze(-1)

    def _pair_forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)

        q = self.norm_in(self.proj(feat_t))
        k = self.norm_in(self.proj(feat_t1))

        if self.self_blocks is not None:
            for block in self.self_blocks:

                def _self_q(
                    nodes: torch.Tensor,
                    mask: torch.Tensor | None,
                    _b: CrossAttentionBlock = block,
                ) -> torch.Tensor:
                    return _b(nodes, nodes, kv_mask=mask)

                def _self_k(
                    nodes: torch.Tensor,
                    mask: torch.Tensor | None,
                    _b: CrossAttentionBlock = block,
                ) -> torch.Tensor:
                    return _b(nodes, nodes, kv_mask=mask)

                if self.gradient_checkpointing and torch.is_grad_enabled():
                    q = grad_ckpt(_self_q, q, mask_t, use_reentrant=False)
                    k = grad_ckpt(_self_k, k, mask_t1, use_reentrant=False)
                else:
                    q = _self_q(q, mask_t)
                    k = _self_k(k, mask_t1)

        for block in self.blocks:

            def _q_fn(
                q: torch.Tensor,
                kv: torch.Tensor,
                mask: torch.Tensor | None,
                _b: CrossAttentionBlock = block,
            ) -> torch.Tensor:
                return _b(q, kv, kv_mask=mask)

            def _k_fn(
                k: torch.Tensor,
                kv: torch.Tensor,
                mask: torch.Tensor | None,
                _b: CrossAttentionBlock = block,
            ) -> torch.Tensor:
                return _b(k, kv, kv_mask=mask)

            if self.gradient_checkpointing and torch.is_grad_enabled():
                q = grad_ckpt(_q_fn, q, k, mask_t1, use_reentrant=False)
                k = grad_ckpt(_k_fn, k, q, mask_t, use_reentrant=False)
            else:
                q = _q_fn(q, k, mask_t1)
                k = _k_fn(k, q, mask_t)

        q = self.norm_out(q)
        k = self.norm_out(k)

        N_t = q.shape[1]
        chunk = self.pair_chunk_size or N_t
        chunks = []

        for i in range(0, N_t, chunk):
            q_c = q[:, i : i + chunk, :]
            coords_c = coords_t[:, i : i + chunk, :]

            def _chunk_fn(
                qc: torch.Tensor,
                kk: torch.Tensor,
                cc: torch.Tensor,
                cc1: torch.Tensor,
                _self: SimpleNodeTransformer = self,
            ) -> torch.Tensor:
                return _self._pair_scores(qc, kk, cc, cc1)

            if self.gradient_checkpointing and torch.is_grad_enabled():
                out = grad_ckpt(_chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False)
            else:
                out = _chunk_fn(q_c, k, coords_c, coords_t1)

            chunks.append(out)

        logits = torch.cat(chunks, dim=1)

        if unbatched:
            logits = logits.squeeze(0)
            q_out = q.squeeze(0)
            k_out = k.squeeze(0)
        else:
            q_out = q
            k_out = k
        return logits, q_out, k_out

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _, _ = self._pair_forward(feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1)
        return logits

    def pair_embeddings(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._pair_forward(feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1)
