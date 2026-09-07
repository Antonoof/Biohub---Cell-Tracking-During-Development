import torch
import torch.nn as nn

from biohub.models.node_transformer import SimpleNodeTransformer


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
    ):
        super().__init__()
        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def index_features(
        self,
        feat_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, C = feat_maps.shape[:2]
        spatial = feat_maps.shape[2:]
        max_nodes = coords.shape[1]

        out = torch.zeros(B, max_nodes, C, device=feat_maps.device, dtype=feat_maps.dtype)
        for b in range(B):
            nt = int(mask[b].sum().item())
            if nt == 0:
                continue
            z = coords[b, :nt, 0].long().clamp(0, spatial[0] - 1)
            y = coords[b, :nt, 1].long().clamp(0, spatial[1] - 1)
            x = coords[b, :nt, 2].long().clamp(0, spatial[2] - 1)
            out[b, :nt] = feat_maps[b, :, z, y, x].T
        return out

    def detect(
        self,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)
        unet_out = self.unet(pair)
        det = self.detect_head(unet_out[0, 0:1])
        return det[0, 0]

    def encode(
        self,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        window = imgs.unsqueeze(2)
        unet_out = self.unet(window)
        W = unet_out.shape[1]
        det_logits = [self.detect_head(unet_out[:, i]) for i in range(W)]
        return unet_out, det_logits

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
