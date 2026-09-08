import math
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class Adan(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
        no_prox: bool = False,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                no_prox=no_prox,
            ),
        )

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)
            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        global_grad_norm.add_(param.grad.pow(2).sum())
            global_grad_norm = torch.sqrt(global_grad_norm)
            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + group['eps']), max=1.0
            ).item()
        else:
            clip_global_grad_norm = 1.0
        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            beta1, beta2, beta3 = group['betas']
            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3_sqrt = math.sqrt(1.0 - beta3 ** group['step'])
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            no_prox = group['no_prox']
            step_size = lr / bias_correction1
            step_size_diff = lr * beta2 / bias_correction2
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad * clip_global_grad_norm
                state = self.state[param]
                if not state:
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                    state['exp_avg_diff'] = torch.zeros_like(param)
                if 'neg_pre_grad' not in state or group['step'] == 1:
                    state['neg_pre_grad'] = grad.clone().mul_(-1.0)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_diff = state['exp_avg_diff']
                neg_grad_or_diff = state['neg_pre_grad']
                neg_grad_or_diff.add_(grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff, alpha=1.0 - beta2)
                neg_grad_or_diff.mul_(beta2).add_(grad)
                exp_avg_sq.mul_(beta3).addcmul_(
                    neg_grad_or_diff, neg_grad_or_diff, value=1.0 - beta3
                )
                denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
                if no_prox:
                    param.mul_(1.0 - lr * weight_decay)
                    param.addcdiv_(exp_avg, denom, value=-step_size)
                    param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
                else:
                    param.addcdiv_(exp_avg, denom, value=-step_size)
                    param.addcdiv_(exp_avg_diff, denom, value=-step_size_diff)
                    param.div_(1.0 + lr * weight_decay)
                neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)
        return loss


class AdamP(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                delta=delta,
                wd_ratio=wd_ratio,
                nesterov=nesterov,
            ),
        )

    def _channel_view(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(value.size(0), -1)

    def _layer_view(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(1, -1)

    def _cosine_similarity(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        eps: float,
        view_func: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        return F.cosine_similarity(view_func(left), view_func(right), dim=1, eps=eps).abs_()

    def _projection(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        perturb: torch.Tensor,
        delta: float,
        wd_ratio: float,
        eps: float,
    ) -> tuple[torch.Tensor, float]:
        wd = 1.0
        expand_size = [-1] + [1] * (len(param.shape) - 1)
        for view_func in (self._channel_view, self._layer_view):
            cosine_sim = self._cosine_similarity(grad, param, eps, view_func)
            if cosine_sim.max() < delta / math.sqrt(view_func(param).size(1)):
                p_n = param / view_func(param).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                return perturb, wd_ratio
        return perturb, wd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']
                state = self.state[param]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                if nesterov:
                    perturb = (beta1 * exp_avg + (1.0 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom
                wd_ratio = 1.0
                if len(param.shape) > 1:
                    perturb, wd_ratio = self._projection(
                        param,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )
                if group['weight_decay'] > 0:
                    param.mul_(1.0 - group['lr'] * group['weight_decay'] * wd_ratio)
                param.add_(perturb, alpha=-step_size)
        return loss


def _zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    if grad.size(-2) > grad.size(-1):
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if grad.size(-2) > grad.size(-1):
        x = x.mT
    return x


def _muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
    return update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5


def _adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    buf1.lerp_(grad, 1.0 - betas[0])
    buf2.lerp_(grad.square(), 1.0 - betas[1])
    buf1c = buf1 / (1.0 - betas[0] ** step)
    buf2c = buf2 / (1.0 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


def _muon_param_groups(
    params: Iterable[object],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    adam_lr: float,
    betas: tuple[float, float],
    eps: float,
) -> list[dict[str, object]]:
    materialized = list(params)
    if materialized and isinstance(materialized[0], dict):
        groups: list[dict[str, object]] = []
        for raw in materialized:
            if not isinstance(raw, dict):
                raise TypeError('Muon param groups must be dicts')
            group: dict[str, object] = raw
            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
            else:
                group['lr'] = group.get('lr', adam_lr)
                group['betas'] = group.get('betas', betas)
                group['eps'] = group.get('eps', eps)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
            groups.append(group)
        return groups
    tensors = [param for param in materialized if isinstance(param, torch.Tensor)]
    muon_params = [param for param in tensors if param.ndim >= 2]
    adam_params = [param for param in tensors if param.ndim < 2]
    groups: list[dict[str, object]] = []
    if muon_params:
        groups.append(
            dict(
                params=muon_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                use_muon=True,
            )
        )
    if adam_params:
        groups.append(
            dict(
                params=adam_params,
                lr=adam_lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                use_muon=False,
            )
        )
    return groups


class MuonWithAuxAdam(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        adam_lr: float | None = None,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-10,
    ) -> None:
        aux_lr = lr if adam_lr is None else adam_lr
        groups = _muon_param_groups(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            adam_lr=aux_lr,
            betas=betas,
            eps=eps,
        )
        super().__init__(groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group['use_muon']:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if not state:
                        state['momentum_buffer'] = torch.zeros_like(param)
                    update = _muon_update(
                        param.grad, state['momentum_buffer'], beta=group['momentum']
                    )
                    param.mul_(1.0 - group['lr'] * group['weight_decay'])
                    param.add_(update.reshape(param.shape), alpha=-group['lr'])
                continue
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                    state['step'] = 0
                state['step'] += 1
                update = _adam_update(
                    param.grad,
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state['step'],
                    group['betas'],
                    group['eps'],
                )
                param.mul_(1.0 - group['lr'] * group['weight_decay'])
                param.add_(update, alpha=-group['lr'])
        return loss
