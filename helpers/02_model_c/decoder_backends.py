"""Sklearn-like pair/source backends for the Model C V2 decoder."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _as_2d(arr: np.ndarray, empty_cols: int = 0) -> np.ndarray:
    a = np.asarray(arr, np.float32)
    if a.ndim == 2:
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == 0:
        return np.zeros((0, empty_cols), np.float32)
    return np.nan_to_num(a.reshape(a.shape[0], -1), nan=0.0, posinf=0.0, neginf=0.0)


def ensure_2d(arr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Force a (n_rows, n_cols) float32 block; empty/1D caches pad with zeros."""
    a = _as_2d(arr, empty_cols=n_cols)
    if a.shape[0] != n_rows:
        if a.size == 0:
            a = np.zeros((n_rows, n_cols), np.float32)
        else:
            raise ValueError(f"row mismatch {a.shape[0]} vs {n_rows}")
    if a.shape[1] != n_cols:
        if a.shape[1] == 0:
            a = np.zeros((a.shape[0], n_cols), np.float32)
        else:
            raise ValueError(f"col mismatch {a.shape[1]} vs {n_cols}")
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def hstack_blocks(parts: list[np.ndarray]) -> np.ndarray:
    cleaned: list[np.ndarray] = []
    n = None
    for p in parts:
        a = _as_2d(p)
        if n is None:
            n = int(a.shape[0])
        elif a.shape[0] != n:
            if a.size == 0:
                a = np.zeros((n, a.shape[1]), np.float32)
            else:
                raise ValueError(f"row mismatch {a.shape[0]} vs {n}")
        cleaned.append(a)
    if not cleaned:
        return np.zeros((0, 0), np.float32)
    return np.concatenate(cleaned, axis=1)


def _zero_last_linear(module: nn.Module) -> None:
    last = None
    for mod in module.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


class ResidualBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + h


class RealMLPNet(nn.Module):
    def __init__(self, n: int, hidden: int = 96, depth: int = 3, dropout: float = 0.08):
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)


class _MLP(nn.Module):
    def __init__(self, n: int, widths: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = n
        for w in widths:
            layers += [nn.Linear(d, w), nn.SiLU(), nn.Dropout(dropout)]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        _zero_last_linear(self.net)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TabMNet(nn.Module):
    def __init__(self, n: int, hidden: int = 64, depth: int = 2, k: int = 4, dropout: float = 0.05):
        super().__init__()
        self.adapters = nn.ModuleList(
            [_MLP(n, [hidden] * depth + [hidden // 2], dropout) for _ in range(k)]
        )

    def forward(self, x):
        return torch.stack([m(x) for m in self.adapters], dim=0).mean(0)


class TorchClassifier:
    """Binary classifier with predict_proba[:, 1] like sklearn."""

    def __init__(self, kind: str, seed: int = 0, device: str = "cpu", loss: str = "bce"):
        self.kind = kind
        self.seed = int(seed)
        self.device = device
        self.loss = loss
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.state_: dict[str, Any] | None = None
        self.n_in_: int | None = None
        self.epochs_ran_ = 0

    def _build(self, n_in: int) -> nn.Module:
        if self.kind == "realmlp":
            return RealMLPNet(n_in)
        if self.kind == "tabm":
            return TabMNet(n_in)
        raise ValueError(self.kind)

    def fit(self, x, y, sample_weight=None):
        x = _as_2d(x)
        y = np.asarray(y, np.float32).ravel()
        if sample_weight is None:
            sample_weight = np.ones(len(y), np.float32)
        sample_weight = np.asarray(sample_weight, np.float32).ravel()
        self.mean_ = x.mean(0).astype(np.float32)
        self.std_ = np.clip(x.std(0).astype(np.float32), 1e-6, None)
        z = (x - self.mean_) / self.std_
        self.n_in_ = int(z.shape[1])
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(self.device if ":" in self.device else "cuda")
        self.device = str(device)
        model = self._build(self.n_in_).to(device)
        xt = torch.from_numpy(z)
        yt = torch.from_numpy(y)
        wt = torch.from_numpy(sample_weight)
        n = len(y)
        n_val = max(1, int(0.12 * n)) if n > 64 else 0
        if n_val:
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            va, tr = perm[:n_val], perm[n_val:]
            train_ds = TensorDataset(xt[tr], yt[tr], wt[tr])
            val_x, val_y = xt[va].to(device), yt[va].to(device)
        else:
            train_ds = TensorDataset(xt, yt, wt)
            val_x = val_y = None
        loader = DataLoader(train_ds, batch_size=min(4096, max(256, n // 8 or 256)), shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best_state = None
        best_val = float("inf")
        patience = 6
        bad = 0
        max_epochs = 30
        for epoch in range(max_epochs):
            model.train()
            for xb, yb, wb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                opt.zero_grad(set_to_none=True)
                logit = model(xb)
                bce = F.binary_cross_entropy_with_logits(logit, yb, reduction="none")
                if self.loss == "focal":
                    pt = torch.exp(-bce)
                    bce = ((1.0 - pt) ** 1.5) * bce
                loss = (wb * bce).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            self.epochs_ran_ = epoch + 1
            if val_x is None:
                continue
            model.eval()
            with torch.no_grad():
                vloss = float(F.binary_cross_entropy_with_logits(model(val_x), val_y).item())
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.state_ = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self._model = model
        return self

    def _ensure(self):
        if getattr(self, "_model", None) is not None:
            return self._model
        device = torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")
        model = self._build(int(self.n_in_)).to(device)
        model.load_state_dict(self.state_)
        model.eval()
        self._model = model
        self.device = str(device)
        return model

    def predict_proba(self, x):
        x = _as_2d(x, empty_cols=int(self.n_in_ or 0))
        if len(x) == 0:
            return np.zeros((0, 2), np.float32)
        z = (x - self.mean_) / self.std_
        model = self._ensure()
        device = next(model.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(z), 8192):
                logit = model(torch.from_numpy(z[i : i + 8192]).to(device))
                p = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
                out.append(p)
        p = np.concatenate(out)
        return np.stack([1.0 - p, p], axis=1).astype(np.float32)

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_model", None)
        return d


class _TreeWrap:
    def __init__(self, impl):
        self.impl = impl

    def fit(self, x, y, sample_weight=None, group=None):
        x = _as_2d(x)
        y = np.asarray(y).ravel()
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight, np.float32)
        self.impl.fit(x, y, **kw)
        return self

    def predict_proba(self, x):
        x = _as_2d(x)
        if len(x) == 0:
            return np.zeros((0, 2), np.float32)
        p = self.impl.predict_proba(x)
        return np.asarray(p, np.float32)


class _RankWrap:
    def __init__(self, impl, kind: str):
        self.impl = impl
        self.kind = kind

    def fit(self, x, y, sample_weight=None, group=None):
        x = _as_2d(x)
        y = np.asarray(y).ravel()
        if group is None:
            raise ValueError(f"{self.kind} ranker requires group sizes")
        group = np.asarray(group, np.int32).ravel()
        if self.kind == "catboost_rank":
            from catboost import Pool

            gid = np.repeat(np.arange(len(group), dtype=np.int32), group)
            pool = Pool(x, y, group_id=gid)
            self.impl.fit(pool)
            return self
        kw = {"group": group.tolist()}
        if sample_weight is not None:
            kw["sample_weight"] = np.asarray(sample_weight, np.float32)
        self.impl.fit(x, y, **kw)
        return self

    def predict_proba(self, x):
        x = _as_2d(x)
        if len(x) == 0:
            return np.zeros((0, 2), np.float32)
        s = np.asarray(self.impl.predict(x), np.float32).ravel()
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -20.0, 20.0)))
        return np.stack([1.0 - p, p], axis=1).astype(np.float32)


def make_backend(name: str, seed: int, device: str = "cpu", role: str = "pair"):
    name = name.strip().lower()
    if name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=280,
            depth=6,
            learning_rate=0.06,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            l2_leaf_reg=3.0,
            thread_count=8,
        )
        return _TreeWrap(model)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=24,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            random_state=seed,
            verbosity=-1,
            n_jobs=8,
        )
        return _TreeWrap(model)
    if name == "catboost_deep":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.04,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            l2_leaf_reg=4.0,
            thread_count=8,
        )
        return _TreeWrap(model)
    if name == "catboost_rank":
        from catboost import CatBoostRanker

        model = CatBoostRanker(
            iterations=300,
            depth=6,
            learning_rate=0.06,
            loss_function="YetiRank",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        )
        return _RankWrap(model, "catboost_rank")
    if name == "lightgbm_rank":
        from lightgbm import LGBMRanker

        model = LGBMRanker(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=16,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            random_state=seed,
            verbosity=-1,
            n_jobs=8,
            objective="lambdarank",
        )
        return _RankWrap(model, "lightgbm_rank")
    if name in {"hgb", "histgb", "sklearn", "hgb_deep", "hgb_slow", "hgb_shallow"}:
        from sklearn.ensemble import HistGradientBoostingClassifier

        if name == "hgb_deep":
            kw = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0)
        elif name == "hgb_slow":
            kw = dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=31, min_samples_leaf=24, l2_regularization=3.0)
        elif name == "hgb_shallow":
            kw = dict(learning_rate=0.08, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=2.0)
        elif role == "source":
            kw = dict(learning_rate=0.055, max_iter=260, max_leaf_nodes=31, min_samples_leaf=24, l2_regularization=2.0)
        else:
            kw = dict(learning_rate=0.065, max_iter=220, max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=1.5)
        return _TreeWrap(HistGradientBoostingClassifier(random_state=seed, **kw))
    if name in {"tabm", "realmlp", "tabm_focal", "realmlp_focal"}:
        kind = name.replace("_focal", "")
        loss = "focal" if name.endswith("_focal") else "bce"
        return TorchClassifier(kind, seed=seed, device=device, loss=loss)
    raise ValueError(f"unknown backend {name}")


def nnls_blend(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights, L2 to labels, then L1-normalize. Fallback: uniform."""
    from scipy.optimize import nnls

    P = np.asarray(P, np.float64)
    y = np.asarray(y, np.float64).ravel()
    if P.ndim != 2 or P.shape[0] != len(y) or P.shape[1] == 0 or P.shape[0] < 2:
        n = max(int(P.shape[1]) if P.ndim == 2 else 1, 1)
        return np.ones(n, np.float64) / n
    w, _ = nnls(P, y)
    s = float(w.sum())
    if s <= 1e-12 or not math.isfinite(s):
        return np.ones(P.shape[1], np.float64) / P.shape[1]
    return w / s
