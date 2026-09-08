import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cellpose.models as cellpose_models  # ty: ignore[unresolved-import]
except ImportError:
    cellpose_models = None

try:
    from segment_anything import sam_model_registry  # ty: ignore[unresolved-import]
except ImportError:
    sam_model_registry = None


class FrozenConvEncoder(nn.Module):
    def __init__(self, out_channels: int, freeze: bool = True) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while groups > 1 and out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, groups), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, groups), out_channels),
        )
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch, time = imgs.shape[:2]
        hidden = self.net(imgs.reshape(batch * time, 1, *imgs.shape[2:]))
        return hidden.reshape(batch, time, *hidden.shape[1:])


class FrozenCellposeEncoder(nn.Module):
    def __init__(self, out_channels: int, freeze: bool = True) -> None:
        super().__init__()
        if cellpose_models is None:
            raise ValueError('extra_encoder=cellpose requires the cellpose package')
        self.model = cellpose_models.CellposeModel(gpu=False)
        self.proj = nn.Conv3d(32, out_channels, kernel_size=1)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch, time = imgs.shape[:2]
        slices = imgs.reshape(batch * time, *imgs.shape[2:])
        depth = slices.shape[1]
        mid = slices[:, depth // 2]
        stacked = mid.unsqueeze(1).expand(-1, depth, -1, -1).reshape(-1, 1, *mid.shape[-2:])
        with torch.no_grad():
            net = self.model.net
            encoded = net(stacked.to(dtype=next(net.parameters()).dtype))
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
        if encoded.ndim == 4:
            channels = encoded.shape[1]
            encoded = encoded.reshape(batch * time, depth, channels, *encoded.shape[-2:])
            encoded = encoded.permute(0, 2, 1, 3, 4)
        encoded = F.interpolate(
            encoded.float(), size=slices.shape[1:], mode='trilinear', align_corners=False
        )
        projected = self.proj(encoded)
        return projected.reshape(batch, time, *projected.shape[1:])


class FrozenSamEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        weights: str | None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if sam_model_registry is None:
            raise ValueError('extra_encoder=sam requires the segment-anything package')
        if not weights:
            raise ValueError('extra_encoder=sam requires extra_encoder_weights')
        sam = sam_model_registry['vit_b'](checkpoint=weights)
        self.image_encoder = sam.image_encoder
        self.proj = nn.Conv3d(256, out_channels, kernel_size=1)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        batch, time = imgs.shape[:2]
        slices = imgs.reshape(batch * time, *imgs.shape[2:])
        depth = slices.shape[1]
        mid = slices[:, depth // 2].unsqueeze(1).repeat(1, 3, 1, 1)
        mid = F.interpolate(mid, size=(1024, 1024), mode='bilinear', align_corners=False)
        with torch.no_grad():
            encoded = self.image_encoder(mid)
        encoded = encoded.mean(dim=(-2, -1), keepdim=True)
        volume = encoded.unsqueeze(2).expand(-1, -1, depth, -1, -1)
        volume = F.interpolate(
            volume.float(), size=slices.shape[1:], mode='trilinear', align_corners=False
        )
        projected = self.proj(volume)
        return projected.reshape(batch, time, *projected.shape[1:])


def make_extra_encoder(
    kind: str,
    out_channels: int,
    *,
    freeze: bool = True,
    weights: str | None = None,
) -> nn.Module | None:
    if kind == 'none':
        return None
    if kind == 'conv':
        return FrozenConvEncoder(out_channels, freeze=freeze)
    if kind == 'cellpose':
        return FrozenCellposeEncoder(out_channels, freeze=freeze)
    if kind == 'sam':
        return FrozenSamEncoder(out_channels, weights, freeze=freeze)
    raise ValueError(f'Unknown extra_encoder {kind!r}')


def coord_channels(
    imgs: torch.Tensor,
    kind: str,
    fourier_bands: int,
) -> torch.Tensor:
    batch, time = imgs.shape[:2]
    depth, height, width = imgs.shape[2:]
    device = imgs.device
    dtype = imgs.dtype
    zz = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype)
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing='ij')
    coords = torch.stack((grid_z, grid_y, grid_x), dim=0)
    if kind == 'coord':
        tiled = coords.unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1, -1, -1)
        return tiled
    if kind == 'fourier':
        bands = 2 ** torch.arange(fourier_bands, device=device, dtype=dtype) * math.pi
        angles = coords.unsqueeze(0) * bands.view(-1, 1, 1, 1, 1)
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=0)
        features = features.reshape(-1, depth, height, width)
        return features.unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1, -1, -1)
    raise ValueError(f'Unknown coord_kind {kind!r}')


def flow_channels(imgs: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == 'frame_diff':
        diff = imgs[:, 1:] - imgs[:, :-1]
        pad = torch.zeros_like(imgs[:, :1])
        return torch.cat((pad, diff), dim=1).unsqueeze(2)
    dz = torch.zeros_like(imgs)
    dy = torch.zeros_like(imgs)
    dx = torch.zeros_like(imgs)
    dz[:, :, 1:] = imgs[:, :, 1:] - imgs[:, :, :-1]
    dy[:, :, :, 1:] = imgs[:, :, :, 1:] - imgs[:, :, :, :-1]
    dx[:, :, :, :, 1:] = imgs[:, :, :, :, 1:] - imgs[:, :, :, :, :-1]
    grad = torch.stack((dz, dy, dx), dim=2)
    if kind == 'spatial_grad':
        return grad
    if kind == 'frame_diff_grad':
        return torch.cat((flow_channels(imgs, 'frame_diff'), grad), dim=2)
    raise ValueError(f'Unknown flow_input {kind!r}')
