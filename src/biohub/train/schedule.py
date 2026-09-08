import math
from collections.abc import Iterable
from copy import deepcopy
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler

from biohub.train.optim import AdamP, Adan, MuonWithAuxAdam


def build_optimizer(
    parameters: Iterable[nn.Parameter] | nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params: Iterable[nn.Parameter]
    if isinstance(parameters, nn.Module):
        params = parameters.parameters()
    else:
        params = parameters
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == 'adan':
        return Adan(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamp':
        return AdamP(params, lr=lr, weight_decay=weight_decay)
    if name == 'muonwithauxadam':
        return MuonWithAuxAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unknown optimizer {name!r}')


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    n_epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> LRScheduler | None:
    if name == 'none':
        return None
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1), eta_min=min_lr)
    if name == 'cosine_warmup':
        warmup = max(int(warmup_epochs), 0)
        floor = min_lr / max(base_lr, 1e-12)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            span = max(n_epochs - warmup, 1)
            progress = (epoch - warmup) / span
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return floor + (1.0 - floor) * cosine

        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(f'Unknown scheduler {name!r}')


TModule = TypeVar('TModule', bound=nn.Module)


class ModelEma(Generic[TModule]):
    def __init__(self, model: TModule, decay: float) -> None:
        self.decay = float(decay)
        shadow = deepcopy(model)
        shadow.eval()
        self.shadow = shadow
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: TModule) -> None:
        d = self.decay
        source = model.state_dict()
        for name, param in self.shadow.state_dict().items():
            value = source[name]
            if param.dtype.is_floating_point:
                param.copy_(param * d + value * (1.0 - d))
            else:
                param.copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow.state_dict()


def normalize_amp(value: object) -> str:
    if value is False or value is None:
        return 'off'
    text = str(value)
    if text in {'False', 'false', 'None', 'none'}:
        return 'off'
    return text


def amp_dtype(name: object) -> torch.dtype | None:
    text = normalize_amp(name)
    if text == 'off':
        return None
    if text == 'fp16':
        return torch.float16
    if text == 'bf16':
        return torch.bfloat16
    raise ValueError(f'Unknown amp {text!r}')
