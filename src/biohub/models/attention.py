from contextlib import AbstractContextManager, nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

FLASH_MIN_SEQ = 32


def divisible_heads(dim: int, n_heads: int) -> int:
    heads = max(int(n_heads), 1)
    width = max(int(dim), 1)
    while heads > 1 and width % heads != 0:
        heads -= 1
    return heads


def attention_kernel(
    seq_len: int,
    device: torch.device | str | None = None,
) -> AbstractContextManager[None]:
    if int(seq_len) >= FLASH_MIN_SEQ:
        return nullcontext()
    if device is None:
        if not torch.cuda.is_available():
            return nullcontext()
    else:
        kind = device.type if isinstance(device, torch.device) else str(device).split(':', 1)[0]
        if kind != 'cuda':
            return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)
