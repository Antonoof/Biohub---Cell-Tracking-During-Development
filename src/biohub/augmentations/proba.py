from typing import Any, Protocol


class AugmentRng(Protocol):
    def random(self, *args: Any, **kwargs: Any) -> Any: ...

    def integers(self, low: Any, high: Any = None, size: Any = None) -> Any: ...

    def normal(self, loc: Any, scale: Any, size: Any) -> Any: ...

    def uniform(self, low: Any, high: Any) -> Any: ...

    def poisson(self, lam: Any) -> Any: ...


def skip_augment(rng: AugmentRng, proba: float) -> bool:
    if proba >= 1.0:
        return False
    if proba <= 0.0:
        return True
    return bool(rng.random() >= proba)
