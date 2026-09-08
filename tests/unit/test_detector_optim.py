import math

import torch
import torch.nn as nn

from biohub.paths import PROJECT_ROOT
from biohub.train.optim import (
    AdamP,
    Adan,
    MuonWithAuxAdam,
    _zeropower_via_newtonschulz5,
)
from biohub.train.schedule import (
    ModelEma,
    amp_dtype,
    build_optimizer,
    build_scheduler,
    normalize_amp,
)
from biohub.utils.yaml_config import load_yaml


def test_amp_yaml_off_is_disabled() -> None:
    assert normalize_amp(False) == 'off'
    assert amp_dtype(False) is None
    assert amp_dtype('off') is None
    cfg = load_yaml(PROJECT_ROOT / 'configs' / '01_p1.yaml')
    assert amp_dtype(cfg['amp']) is None


def test_adamw_step_decreases_loss() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = build_optimizer(model.parameters(), name='adamw', lr=0.1, weight_decay=0.0)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    before = ((model(x) - y) ** 2).mean()
    before.backward()
    opt.step()
    after = ((model(x) - y) ** 2).mean()
    assert float(after) < float(before)


def test_cosine_reaches_min_lr() -> None:
    model = nn.Linear(2, 2)
    opt = build_optimizer(model.parameters(), name='sgd', lr=1.0, weight_decay=0.0)
    sched = build_scheduler(
        opt, name='cosine', n_epochs=2, warmup_epochs=0, min_lr=0.0, base_lr=1.0
    )
    assert sched is not None
    sched.step()
    sched.step()
    assert opt.param_groups[0]['lr'] <= 1e-6


def test_cosine_warmup_starts_below_base() -> None:
    model = nn.Linear(2, 2)
    opt = build_optimizer(model.parameters(), name='sgd', lr=1.0, weight_decay=0.0)
    sched = build_scheduler(
        opt, name='cosine_warmup', n_epochs=4, warmup_epochs=2, min_lr=0.0, base_lr=1.0
    )
    assert sched is not None
    assert opt.param_groups[0]['lr'] == 0.5
    sched.step()
    assert abs(opt.param_groups[0]['lr'] - 1.0) < 1e-6


def test_ema_updates_shadow() -> None:
    torch.manual_seed(0)
    model = nn.Linear(2, 2, bias=False)
    ema = ModelEma(model, decay=0.5)
    before = ema.shadow.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert not torch.equal(ema.shadow.weight, before)
    expected = before * 0.5 + model.weight * 0.5
    torch.testing.assert_close(ema.shadow.weight, expected)


def test_accum_two_steps_match_mean_loss() -> None:
    torch.manual_seed(0)
    model_a = nn.Linear(3, 1, bias=False)
    torch.manual_seed(0)
    model_b = nn.Linear(3, 1, bias=False)
    x1 = torch.randn(2, 3)
    x2 = torch.randn(2, 3)
    y1 = torch.randn(2, 1)
    y2 = torch.randn(2, 1)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
    loss = (((model_a(x1) - y1) ** 2).mean() + ((model_a(x2) - y2) ** 2).mean()) / 2
    loss.backward()
    opt_a.step()
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    ((model_b(x1) - y1) ** 2).mean().div(2).backward()
    ((model_b(x2) - y2) ** 2).mean().div(2).backward()
    opt_b.step()
    torch.testing.assert_close(model_a.weight, model_b.weight)


def test_adan_adamp_muon_decrease_loss() -> None:
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    for name in ('adan', 'adamp', 'muonwithauxadam'):
        torch.manual_seed(1)
        model = nn.Linear(4, 1)
        opt = build_optimizer(model, name=name, lr=0.05, weight_decay=0.0)
        before = ((model(x) - y) ** 2).mean()
        before.backward()
        opt.step()
        after = ((model(x) - y) ** 2).mean()
        assert float(after) <= float(before) + 1e-5


def _sail_adan_step(
    param: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    step: int,
    lr: float,
    betas: tuple[float, float, float],
    eps: float,
    weight_decay: float,
    no_prox: bool,
) -> None:
    grad = param.grad
    assert grad is not None
    beta1, beta2, beta3 = betas
    if 'exp_avg' not in state:
        state['exp_avg'] = torch.zeros_like(param)
        state['exp_avg_sq'] = torch.zeros_like(param)
        state['exp_avg_diff'] = torch.zeros_like(param)
    if 'pre_grad' not in state:
        diff = torch.zeros_like(grad)
    else:
        diff = grad - state['pre_grad']
    state['exp_avg'].mul_(beta1).add_(grad, alpha=1.0 - beta1)
    state['exp_avg_diff'].mul_(beta2).add_(diff, alpha=1.0 - beta2)
    nesterov_grad = grad + beta2 * diff
    state['exp_avg_sq'].mul_(beta3).addcmul_(nesterov_grad, nesterov_grad, value=1.0 - beta3)
    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(1.0 - beta3**step)).add_(eps)
    if no_prox:
        param.mul_(1.0 - lr * weight_decay)
        param.addcdiv_(state['exp_avg'], denom, value=-lr / (1.0 - beta1**step))
        param.addcdiv_(state['exp_avg_diff'], denom, value=-lr * beta2 / (1.0 - beta2**step))
    else:
        param.addcdiv_(state['exp_avg'], denom, value=-lr / (1.0 - beta1**step))
        param.addcdiv_(state['exp_avg_diff'], denom, value=-lr * beta2 / (1.0 - beta2**step))
        param.div_(1.0 + lr * weight_decay)
    state['pre_grad'] = grad.clone()


def test_adan_matches_sail_sg() -> None:
    torch.manual_seed(0)
    ours = nn.Linear(6, 4)
    torch.manual_seed(0)
    ref = nn.Linear(6, 4)
    x = torch.randn(8, 6)
    y = torch.randn(8, 4)
    betas = (0.98, 0.92, 0.99)
    opt = Adan(ours.parameters(), lr=0.01, betas=betas, eps=1e-8, weight_decay=0.02)
    states: dict[str, dict[str, torch.Tensor]] = {'weight': {}, 'bias': {}}
    for step in range(1, 4):
        ours.zero_grad()
        ref.zero_grad()
        ((ours(x) - y) ** 2).mean().backward()
        ((ref(x) - y) ** 2).mean().backward()
        opt.step()
        with torch.no_grad():
            _sail_adan_step(
                ref.weight,
                states['weight'],
                step=step,
                lr=0.01,
                betas=betas,
                eps=1e-8,
                weight_decay=0.02,
                no_prox=False,
            )
            _sail_adan_step(
                ref.bias,
                states['bias'],
                step=step,
                lr=0.01,
                betas=betas,
                eps=1e-8,
                weight_decay=0.02,
                no_prox=False,
            )
        torch.testing.assert_close(ours.weight, ref.weight)
        torch.testing.assert_close(ours.bias, ref.bias)


def test_adamp_1d_matches_adam() -> None:
    torch.manual_seed(0)
    ours = nn.Parameter(torch.randn(7))
    torch.manual_seed(0)
    ref = nn.Parameter(torch.randn(7))
    opt = AdamP([ours], lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    adam = torch.optim.Adam([ref], lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for _ in range(4):
        grad = torch.randn(7)
        ours.grad = grad.clone()
        ref.grad = grad.clone()
        opt.step()
        adam.step()
    torch.testing.assert_close(ours, ref)


def test_adamp_projects_like_clovaai() -> None:
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(3, 5))
    grad = torch.randn(3, 5)
    before = param.detach().clone()
    param.grad = grad.clone()
    opt = AdamP(
        [param],
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        delta=1e9,
        wd_ratio=0.1,
    )
    opt.step()
    exp_avg = (1.0 - 0.9) * grad
    exp_avg_sq = (1.0 - 0.999) * grad * grad
    denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - 0.999)).add(1e-8)
    perturb = exp_avg / denom
    expand = [-1] + [1] * (len(before.shape) - 1)
    p_n = before / before.view(before.size(0), -1).norm(dim=1).view(expand).add(1e-8)
    perturb = perturb - p_n * (p_n * perturb).view(p_n.size(0), -1).sum(dim=1).view(expand)
    expected = before * (1.0 - 0.1 * 0.01 * 0.1) - (0.1 / (1.0 - 0.9)) * perturb
    torch.testing.assert_close(param.detach(), expected, rtol=1e-5, atol=1e-5)


def test_muon_aux_adam_matches_adam() -> None:
    torch.manual_seed(0)
    ours = nn.Parameter(torch.randn(5))
    torch.manual_seed(0)
    ref = nn.Parameter(torch.randn(5))
    opt = MuonWithAuxAdam([ours], lr=0.03, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-10)
    adam = torch.optim.Adam([ref], lr=0.03, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0)
    for _ in range(4):
        grad = torch.randn(5)
        ours.grad = grad.clone()
        ref.grad = grad.clone()
        opt.step()
        adam.step()
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-6)


def test_muon_matches_kellerjordan() -> None:
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(8, 6))
    grad = torch.randn(8, 6)
    param.grad = grad.clone()
    before = param.detach().clone()
    opt = MuonWithAuxAdam([param], lr=0.05, weight_decay=0.01, momentum=0.95)
    opt.step()
    momentum = (1.0 - 0.95) * grad
    update = grad.lerp(momentum, 0.95)
    ns = _zeropower_via_newtonschulz5(update, steps=5)
    ns = ns * max(1.0, ns.size(-2) / ns.size(-1)) ** 0.5
    expected = before * (1.0 - 0.05 * 0.01)
    expected.add_(ns.reshape(before.shape), alpha=-0.05)
    torch.testing.assert_close(param.detach(), expected, rtol=1e-4, atol=1e-4)


def test_muon_conv4d_view() -> None:
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(4, 3, 3, 3))
    grad = torch.randn_like(param)
    param.grad = grad.clone()
    before = param.detach().clone()
    opt = MuonWithAuxAdam([param], lr=0.02, weight_decay=0.0, momentum=0.95)
    opt.step()
    momentum = (1.0 - 0.95) * grad
    update = grad.lerp(momentum, 0.95).view(4, -1)
    ns = _zeropower_via_newtonschulz5(update, steps=5)
    ns = ns * max(1.0, ns.size(-2) / ns.size(-1)) ** 0.5
    expected = before.clone()
    expected.add_(ns.reshape(before.shape), alpha=-0.02)
    torch.testing.assert_close(param.detach(), expected, rtol=1e-4, atol=1e-4)
