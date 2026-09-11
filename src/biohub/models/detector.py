import torch
import torch.nn as nn
import torch.nn.functional as F

from biohub.models.extra_encoder import (
    coord_channels,
    flow_channels,
    make_extra_encoder,
)
from biohub.models.node_transformer import SimpleNodeTransformer
from biohub.models.temporal_unet import pack_conv3d_channels_last


def _index_nearest(
    feat_maps: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    batch, channels, depth, height, width = feat_maps.shape
    z = coords[..., 0].long().clamp(0, depth - 1)
    y = coords[..., 1].long().clamp(0, height - 1)
    x = coords[..., 2].long().clamp(0, width - 1)
    index = (z * height + y) * width + x
    if feat_maps.is_contiguous(memory_format=torch.channels_last_3d):
        spatial = feat_maps.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        gathered = spatial.gather(1, index.unsqueeze(-1).expand(-1, -1, channels))
        return gathered.masked_fill(~mask[..., None], 0)
    gathered = feat_maps.reshape(batch, channels, -1).gather(
        2, index.unsqueeze(1).expand(-1, channels, -1)
    )
    return gathered.transpose(1, 2).masked_fill(~mask[..., None], 0)


def _index_trilinear(
    feat_maps: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    B, C = feat_maps.shape[:2]
    depth, height, width = feat_maps.shape[2:]
    max_nodes = coords.shape[1]
    out = torch.zeros(B, max_nodes, C, device=feat_maps.device, dtype=feat_maps.dtype)
    if max_nodes == 0:
        return out
    z = coords[..., 0].float().clamp(0, max(depth - 1, 0))
    y = coords[..., 1].float().clamp(0, max(height - 1, 0))
    x = coords[..., 2].float().clamp(0, max(width - 1, 0))
    zn = 2.0 * z / max(depth - 1, 1) - 1.0
    yn = 2.0 * y / max(height - 1, 1) - 1.0
    xn = 2.0 * x / max(width - 1, 1) - 1.0
    grid = torch.stack((xn, yn, zn), dim=-1).view(B, 1, 1, max_nodes, 3)
    sampled = F.grid_sample(
        feat_maps.float(),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    sampled = sampled.view(B, C, max_nodes).transpose(1, 2).to(dtype=feat_maps.dtype)
    return sampled * mask.unsqueeze(-1).to(dtype=sampled.dtype)


class UNetNodeTransformer(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        unet_out_channels: int,
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
        mlp_ratio: float = 2.0,
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
        feature_sample: str = 'nearest',
        coord_kind: str = 'none',
        fourier_bands: int = 4,
        flow_input: str = 'none',
        extra_encoder: str = 'none',
        extra_encoder_channels: int = 8,
        extra_encoder_freeze: bool = True,
        extra_encoder_weights: str | None = None,
    ):
        super().__init__()
        if feature_sample not in ('nearest', 'trilinear'):
            raise ValueError(f'Unknown feature_sample {feature_sample!r}')
        self.unet = unet
        self.unet_out_channels = unet_out_channels
        self.feature_sample = feature_sample
        self.coord_kind = coord_kind
        self.fourier_bands = int(fourier_bands)
        self.flow_input = flow_input
        self.extra_encoder = make_extra_encoder(
            extra_encoder,
            extra_encoder_channels,
            freeze=extra_encoder_freeze,
            weights=extra_encoder_weights,
        )
        self._coord_key: tuple | None = None
        self._coord_cache: torch.Tensor | None = None

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            pair_chunk_size=pair_chunk_size,
            drop_path=drop_path,
            use_self_attn=use_self_attn,
            norm=norm,
            rel_coord_scale=rel_coord_scale,
            pair_head=pair_head,
            layer_scale_init=layer_scale_init,
            gradient_checkpointing=gradient_checkpointing,
            ffn_act=ffn_act,
            attn_dropout=attn_dropout,
            drop_path_decay=drop_path_decay,
            pair_geom=pair_geom,
        )
        self.offset_head = nn.Conv3d(unet_out_channels, 3, kernel_size=1)
        self._heads_channels_last = False
        self.register_buffer(
            '_arch',
            torch.tensor([int(hidden_dim), int(n_heads), int(n_blocks)], dtype=torch.int64),
            persistent=True,
        )

    def _unet_input(self, imgs: torch.Tensor) -> torch.Tensor:
        window = imgs.unsqueeze(2)
        parts = [window]
        if self.coord_kind != 'none':
            key = (
                imgs.shape[0],
                imgs.shape[1],
                *imgs.shape[2:],
                imgs.device,
                imgs.dtype,
                self.coord_kind,
                self.fourier_bands,
            )
            if self._coord_cache is None or self._coord_key != key:
                self._coord_cache = coord_channels(imgs, self.coord_kind, self.fourier_bands)
                self._coord_key = key
            parts.append(self._coord_cache)
        if self.flow_input != 'none':
            parts.append(flow_channels(imgs, self.flow_input))
        if self.extra_encoder is not None:
            parts.append(self.extra_encoder(imgs))
        return torch.cat(parts, dim=2)

    def index_features(
        self,
        feat_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.feature_sample == 'trilinear':
            return _index_trilinear(feat_maps, coords, mask)
        return _index_nearest(feat_maps, coords, mask)

    def detect(
        self,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        imgs = torch.stack([frame, frame], dim=0).unsqueeze(0)
        _unet_out, stacked = self.encode_stacked(imgs)
        return stacked[0, 0, 0]

    def encode_stacked(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = self._unet_input(imgs)
        unet = getattr(self.unet, 'module', self.unet)
        if (
            imgs.is_cuda
            and getattr(unet, '_channels_last', False)
            and not self._heads_channels_last
        ):
            pack_conv3d_channels_last(self.detect_head)
            pack_conv3d_channels_last(self.offset_head)
            self._heads_channels_last = True
        unet_out = self.unet(window)
        batch, frames, channels = unet_out.shape[:3]
        logits = self.detect_head(unet_out.reshape(batch * frames, channels, *unet_out.shape[3:]))
        return unet_out, logits.reshape(batch, frames, *logits.shape[1:])

    def encode(
        self,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        unet_out, stacked = self.encode_stacked(imgs)
        return unet_out, list(stacked.unbind(1))

    def predict_edges(
        self,
        unet_feat_src: torch.Tensor,
        unet_feat_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        pos_feat_src: torch.Tensor,
        pos_feat_tgt: torch.Tensor,
        mask_src: torch.Tensor,
        mask_tgt: torch.Tensor,
    ) -> torch.Tensor:
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer(
            feat_src,
            feat_tgt,
            coords_src,
            coords_tgt,
            mask_src,
            mask_tgt,
        )

    def predict_edges_embeddings(
        self,
        unet_feat_src: torch.Tensor,
        unet_feat_tgt: torch.Tensor,
        coords_src: torch.Tensor,
        coords_tgt: torch.Tensor,
        pos_feat_src: torch.Tensor,
        pos_feat_tgt: torch.Tensor,
        mask_src: torch.Tensor,
        mask_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer.pair_embeddings(
            feat_src,
            feat_tgt,
            coords_src,
            coords_tgt,
            mask_src,
            mask_tgt,
        )
