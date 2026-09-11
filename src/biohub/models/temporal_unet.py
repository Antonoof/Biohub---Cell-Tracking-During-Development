from collections.abc import Sequence
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from biohub.models.attention import divisible_heads


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(self.pool(x).flatten(1)).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * weight


def make_unet_norm(kind: str, channels: int, gn_groups: int) -> nn.Module:
    if kind == 'groupnorm':
        groups = min(int(gn_groups), channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(max(1, groups), channels)
    if kind == 'batchnorm':
        return nn.BatchNorm3d(channels)
    raise ValueError(f'Unknown unet_norm {kind!r}')


def _identity_grid(volume: torch.Tensor) -> torch.Tensor:
    n, _c, depth, height, width = volume.shape
    zz = torch.linspace(-1.0, 1.0, depth, device=volume.device, dtype=volume.dtype)
    yy = torch.linspace(-1.0, 1.0, height, device=volume.device, dtype=volume.dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=volume.device, dtype=volume.dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing='ij')
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
    return grid.unsqueeze(0).expand(n, -1, -1, -1, -1)


class DeformConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.offset = nn.Conv3d(in_channels, 3, kernel_size=1)
        nn.init.zeros_(self.offset.weight)
        if self.offset.bias is not None:
            nn.init.zeros_(self.offset.bias)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self._grid: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self._grid_key: tuple | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        n, _c, depth, height, width = x.shape
        key = (n, depth, height, width, x.device, x.dtype)
        grid = self._grid
        scale = self._scale
        if grid is None or scale is None or self._grid_key != key:
            # Cache may first be populated by validation/EMA. Training must be able
            # to save scale for division backward after leaving inference_mode.
            with torch.inference_mode(False):
                grid = _identity_grid(x)
                scale = torch.tensor(
                    [max(width - 1, 1) / 2.0, max(height - 1, 1) / 2.0, max(depth - 1, 1) / 2.0],
                    device=x.device,
                    dtype=x.dtype,
                ).view(1, 3, 1, 1, 1)
            self._grid = grid
            self._scale = scale
            self._grid_key = key
        warped = grid + (offset.permute(0, 2, 3, 4, 1) / scale.permute(0, 2, 3, 4, 1))
        sampled = F.grid_sample(
            x.float(),
            warped.float(),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return self.conv(sampled.to(dtype=x.dtype))


def pack_conv3d_channels_last(module: nn.Module) -> None:
    skip = {
        id(param)
        for child in module.modules()
        if type(child).__name__ == '_TemporalConv'
        for param in child.parameters()
    }
    with torch.no_grad():
        for param in module.parameters():
            if id(param) in skip:
                continue
            if param.ndim == 5 and param.is_cuda:
                param.data = param.data.contiguous(memory_format=torch.channels_last_3d)


def _spatial_conv(
    in_channels: int,
    out_channels: int,
    *,
    deform: bool,
    kernel_size: int = 3,
    padding: int = 1,
    groups: int = 1,
) -> nn.Module:
    if deform:
        return DeformConv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups
        )
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias=False,
        groups=groups,
    )


def _conv_block(
    in_channels: int,
    out_channels: int,
    se_ratio: float = 0.0,
    *,
    unet_norm: str = 'batchnorm',
    gn_groups: int = 8,
    deform: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        _spatial_conv(in_channels, out_channels, deform=deform),
        make_unet_norm(unet_norm, out_channels, gn_groups),
        nn.ReLU(inplace=True),
        _spatial_conv(out_channels, out_channels, deform=deform),
        make_unet_norm(unet_norm, out_channels, gn_groups),
        nn.ReLU(inplace=True),
    ]
    if se_ratio > 0:
        layers.append(SqueezeExcite(out_channels, se_ratio))
    return nn.Sequential(*layers)


class ResidualBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        unet_norm: str,
        gn_groups: int,
        deform: bool,
    ) -> None:
        super().__init__()
        self.conv1 = _spatial_conv(in_channels, out_channels, deform=deform)
        self.norm1 = make_unet_norm(unet_norm, out_channels, gn_groups)
        self.conv2 = _spatial_conv(out_channels, out_channels, deform=deform)
        self.norm2 = make_unet_norm(unet_norm, out_channels, gn_groups)
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.se = SqueezeExcite(out_channels, se_ratio) if se_ratio > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.relu(x + residual)
        return self.se(x)


class ConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        unet_norm: str,
        gn_groups: int,
        deform: bool,
    ) -> None:
        super().__init__()
        hidden = 4 * in_channels
        self.dw = _spatial_conv(
            in_channels, in_channels, deform=deform, kernel_size=7, padding=3, groups=in_channels
        )
        self.norm = make_unet_norm(unet_norm, in_channels, gn_groups)
        self.pw1 = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, out_channels, kernel_size=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.se = SqueezeExcite(out_channels, se_ratio) if se_ratio > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        return self.se(x + residual)


def make_stage_block(
    in_channels: int,
    out_channels: int,
    *,
    unet_block: str,
    se_ratio: float,
    unet_norm: str,
    gn_groups: int,
    deform: bool,
) -> nn.Module:
    if unet_block == 'plain':
        return _conv_block(
            in_channels,
            out_channels,
            se_ratio,
            unet_norm=unet_norm,
            gn_groups=gn_groups,
            deform=deform,
        )
    if unet_block == 'residual':
        return ResidualBlock3d(in_channels, out_channels, se_ratio, unet_norm, gn_groups, deform)
    if unet_block == 'convnext':
        return ConvNeXtBlock3d(in_channels, out_channels, se_ratio, unet_norm, gn_groups, deform)
    raise ValueError(f'Unknown unet_block {unet_block!r}')


class _TemporalIdentity(nn.Module):
    def forward(self, x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        return x


class _TemporalAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        heads = divisible_heads(channels, n_heads)
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        spatial = x.shape[2:]
        h = x
        if (
            h.is_cuda
            and not h.is_contiguous(memory_format=torch.channels_last_3d)
            and self.qkv.weight.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            h = h.contiguous(memory_format=torch.channels_last_3d)
        h = self.norm(h.movedim(1, -1)).movedim(-1, 1)
        qkv = self.qkv(h).movedim(1, -1).reshape(batch, frames, -1, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(3)
        query = query.permute(0, 3, 2, 1, 4)
        key = key.permute(0, 3, 2, 1, 4)
        value = value.permute(0, 3, 2, 1, 4)
        attn = (query @ key.transpose(-1, -2) * self.scale).softmax(dim=-1)
        out = (attn @ value).permute(0, 3, 2, 1, 4)
        out = out.reshape(batch * frames, *spatial, self.heads * self.head_dim)
        out = self.proj(out.movedim(-1, 1))
        if out.is_cuda and not out.is_contiguous(memory_format=torch.channels_last_3d):
            out = out.contiguous(memory_format=torch.channels_last_3d)
        return x + out


class _TemporalConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        _, channels, depth, height, width = x.shape
        h = x.reshape(batch, frames, channels, depth, height, width)
        h = h.permute(0, 3, 2, 1, 4, 5).reshape(batch * depth, channels, frames, height, width)
        h = self.conv(self.norm(h))
        h = h.reshape(batch, depth, channels, frames, height, width)
        h = h.permute(0, 3, 2, 1, 4, 5).reshape_as(x)
        if x.is_cuda and x.is_contiguous(memory_format=torch.channels_last_3d):
            h = h.contiguous(memory_format=torch.channels_last_3d)
        return x + h


class _TemporalStack(nn.Module):
    def __init__(self, *blocks: nn.Module) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, batch, frames)
        return x


def make_temporal_block(
    channels: int,
    *,
    temporal_mix: str,
    n_heads: int,
    skip: bool,
) -> nn.Module:
    if skip or temporal_mix == 'none':
        return _TemporalIdentity()
    if temporal_mix == 'attn':
        return _TemporalAttention(channels, n_heads=n_heads)
    if temporal_mix == 'conv':
        return _TemporalConv(channels)
    if temporal_mix == 'both':
        return _TemporalStack(
            _TemporalConv(channels), _TemporalAttention(channels, n_heads=n_heads)
        )
    raise ValueError(f'Unknown temporal_mix {temporal_mix!r}')


def unet_in_channels(
    *,
    coord_kind: str = 'none',
    fourier_bands: int = 4,
    flow_input: str = 'none',
    extra_encoder: str = 'none',
    extra_encoder_channels: int = 8,
) -> int:
    channels = 1
    if coord_kind == 'coord':
        channels += 3
    elif coord_kind == 'fourier':
        channels += 6 * int(fourier_bands)
    elif coord_kind != 'none':
        raise ValueError(f'Unknown coord_kind {coord_kind!r}')
    if flow_input == 'frame_diff':
        channels += 1
    elif flow_input == 'spatial_grad':
        channels += 3
    elif flow_input == 'frame_diff_grad':
        channels += 4
    elif flow_input != 'none':
        raise ValueError(f'Unknown flow_input {flow_input!r}')
    if extra_encoder != 'none':
        channels += int(extra_encoder_channels)
    return channels


@contextmanager
def preserve_batchnorm_stats(block: nn.Module):
    states = []
    for m in block.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            for buffer in (m.running_mean, m.running_var, m.num_batches_tracked):
                if buffer is not None:
                    states.append((buffer, buffer.clone()))
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in states:
                buffer.copy_(saved)


class TemporalUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        layers: Sequence[int] = (32, 64, 128),
        gradient_checkpointing: bool = False,
        skip_fullres_temporal: bool = True,
        temporal_n_heads: int = 4,
        se_ratio: float = 0.0,
        unet_block: str = 'plain',
        unet_norm: str = 'batchnorm',
        unet_gn_groups: int = 8,
        unet_deform: bool = False,
        temporal_mix: str = 'attn',
    ) -> None:
        super().__init__()
        stage_widths = list(layers)
        if len(stage_widths) < 2:
            raise ValueError('layers must contain at least two stages')

        self.gradient_checkpointing = gradient_checkpointing

        self.encoder_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(stage_widths):
            self.encoder_blocks.append(
                make_stage_block(
                    prev,
                    ch,
                    unet_block=unet_block,
                    se_ratio=se_ratio,
                    unet_norm=unet_norm,
                    gn_groups=unet_gn_groups,
                    deform=unet_deform,
                )
            )
            self.temporal_blocks.append(
                make_temporal_block(
                    ch,
                    temporal_mix=temporal_mix,
                    n_heads=temporal_n_heads,
                    skip=bool(skip_fullres_temporal and i == 0),
                )
            )
            prev = ch
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(stage_widths) - 1, 0, -1):
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            )
            self.decoder_blocks.append(
                make_stage_block(
                    stage_widths[i] + stage_widths[i - 1],
                    stage_widths[i - 1],
                    unet_block=unet_block,
                    se_ratio=se_ratio,
                    unet_norm=unet_norm,
                    gn_groups=unet_gn_groups,
                    deform=unet_deform,
                )
            )

        self.head = nn.Conv3d(stage_widths[0], out_channels, kernel_size=1)
        self._channels_last = not bool(unet_deform)
        self._channels_last_applied = False

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return grad_ckpt(
                block,
                x,
                use_reentrant=False,
                context_fn=lambda: (nullcontext(), preserve_batchnorm_stats(block)),
            )
        return block(x)

    def _as_channels_last(self, volume: torch.Tensor) -> torch.Tensor:
        if not (self._channels_last and volume.is_cuda):
            return volume
        if not self._channels_last_applied:
            pack_conv3d_channels_last(self)
            self._channels_last_applied = True
        if volume.is_contiguous(memory_format=torch.channels_last_3d):
            return volume
        return volume.contiguous(memory_format=torch.channels_last_3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        x = self._as_channels_last(x.reshape(B * T, *x.shape[2:]))

        skips: list[torch.Tensor] = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if i > 0:
                x = self.pool(x)
            x = self._run(block, x)
            x = temporal(x, B, T)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
            x = up(x.contiguous())
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = self._as_channels_last(
                torch.cat([self._as_channels_last(x), self._as_channels_last(skip)], dim=1)
            )
            x = self._run(block, x)

        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])
